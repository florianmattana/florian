---
title: Designing an FP4 Fused Attention Kernel for SM120
description: Design document for a fused attention kernel targeting consumer Blackwell GPUs
header: FP4 Fused Attention Kernel for SM120
---

## 1. Problem Statement

The efficiency of attention computation is critical for ML inference on consumer hardware, given that its computational cost grows quadratically with sequence length: doubling the number of tokens quadruples the size of the attention score matrix and the number of operations required.

Recent work has demonstrated that FP4 Tensor Cores on Blackwell GPUs can deliver up to 5x speedup over FP16 attention through quantized fused kernels. However, existing implementations target a narrow hardware slice. SageAttention3 (Tsinghua, NeurIPS 2025) achieves 1038 TOPS but has been developed and validated exclusively on the RTX 5090 (GB202, 170 SMs). FlashAttention-4 targets Blackwell datacenter GPUs (SM100). The broader SM120 consumer lineup (RTX 5080, RTX 5070 Ti, RTX 5070, RTX 5060 Ti) remains without a functional FP4 fused attention kernel. Community reports confirm that SageAttention3 crashes on the RTX 5070 Ti and fails to compile on the RTX 5060 Ti, leaving these GPUs unable to exploit their native FP4 Tensor Core capabilities.

This represents a significant missed opportunity. On a RTX 5070 Ti (GB205, 46 SMs), the FP4 Tensor Cores deliver approximately 474 dense TFLOPS, a 4x throughput advantage over FP16 Tensor (123.5 TFLOPS) and 2x over INT8 Tensor (246.9 TFLOPS). Simultaneously, the 4-bit data format provides a 4x reduction in memory traffic compared to FP16, directly alleviating the primary bottleneck on memory-bandwidth-constrained consumer GPUs (672 GB/s on RTX 5070 Ti versus 1,792 GB/s on RTX 5090). The alternative, non-fused FP4 GEMM kernels such as VincentKaufmann's fp4-cuda-kernel (143 TFLOPS on SM120), requires materializing the full N×N attention score matrix in VRAM between the QKᵀ and softmax·V stages, which is prohibitive for long sequences on 12 GB devices.

I design and implement an FP4 fused attention kernel specifically targeting the SM120 consumer GPU family beyond the RTX 5090. The kernel computes softmax(QKᵀ/√d)·V in a single pass using FP4 E2M1 quantization with hardware-native block scaling (one FP16 scale factor per 32 elements), online softmax for numerical stability, and tiled shared memory management optimized for the 128 KB per-SM shared memory and 12 GB VRAM constraints of consumer Blackwell GPUs.

---
## 2. Design Decision: Instruction Path for FP4 Tensor Cores


The three options
Before writing any kernel code, the first architectural decision was to determine how to invoke the FP4 Tensor Cores. Three options were evaluated.

Option A consists of writing the matrix multiplications in inline MMA PTX, providing full control over register placement, shared memory access patterns, and the compute pipeline. The cost is complexity: there is no C++ abstraction layer, and the developer is responsible for encoding operands into the correct register layout expected by the hardware.

Option B consists of using CuTe (the layout and data movement sublibrary within CUTLASS 3.x) to handle operand loading and memory layouts, while writing the fused attention loop and online softmax manually. This reduces the burden of register management but introduces a dependency on CUTLASS internals, and the interaction between CuTe's tile abstractions and a custom fused loop with softmax in the middle is not straightforward.

Option C consists of starting from an existing fused INT8 attention kernel (specifically, the ParagEkbote/model-kernels kernel that I have previously contributed to) and surgically replacing the WMMA INT8 matrix multiplications with FP4 equivalents. This minimizes risk since the tiling, online softmax, and memory management are already proven, but it limits the scope ot the experience to a surface-level swap without deep knowledge of what the hardware is actually doing.

Why Option A
The core constraint of a fused attention kernel is that the intermediate attention score matrix from Q·Kᵀ must never be materialized in VRAM. It must stay in shared memory and registers between the first matrix multiplication (Q·Kᵀ), the online softmax, and the second matrix multiplication (scores·V). Any approach that delegates the matrix multiplications to an external library risks breaking this fusion by requiring data to exit and re-enter the kernel's local memory hierarchy.

Option B was considered but set aside: CuTe is designed around standalone GEMM operations, and inserting a custom softmax between two CuTe-managed matrix multiplications would require fighting the abstraction rather than leveraging it.

Option C was rejected on principle. The goal of this project is not to produce a working kernel by the shortest path, but to share the full stack from PTX instructions to fused attention, and to document the difficulties encountered along the way. Starting from a working kernel and patching it would shortcut that journey.

Option A was selected: CUDA C++ for the kernel structure (launch configuration, shared memory allocation, tiling loop, online softmax, output writeback) with inline PTX assembly specifically for the Tensor Core matrix multiply instructions.

The first question: which PTX instruction activates FP4 Tensor Cores?
With Option A selected, the immediate problem was to identify which PTX instruction to use. The working hypothesis was that all Blackwell GPUs share the same Tensor Core instruction set. The most documented path for FP4 on Blackwell is tcgen05.mma, a 5th generation Tensor Core instruction that operates on Tensor Memory (TMEM), supports massive MMA shapes up to m128×n256×k16, and is the foundation of high-performance datacenter kernels like SageAttention3 and DeepGEMM.

This hypothesis turned out to be wrong.

SM100 vs SM120: a critical architectural split
SM100 (datacenter Blackwell: B200, B300) and SM120 (consumer Blackwell: RTX 5070, 5070 Ti, 5080, 5090) do not share the same instruction set for Tensor Core operations. Three independent sources confirm this.

CUTLASS issue #2800 states explicitly: "FP4 on SM120/SM121 is usable through Ampere-style mma instructions rather than tcgen05." The NVIDIA developer forum post on FP4 performance on DGX Spark identifies the specific instructions: "OMMA and QMMA are warp-level MMA instructions that work on SM120/SM121. These are the tensor core instructions that CUTLASS accesses through CuTe." CUTLASS issue #3044 reveals the exact PTX syntax: mma.sync.aligned.kind::f8f6f4.m16n8k32 when targeting sm_120a.

What SM120 does not have
SM120 lacks several features that SM100 relies on for peak FP4 performance. There is no tcgen05.mma instruction. There is no Tensor Memory (TMEM), the dedicated 128×512 accumulator memory that SM100 uses to reduce register pressure. There is no multicast and no CTA pairs, meaning no cooperative 2-SM execution. The cluster size is fixed at 1×1×1. Shared memory per SM is approximately 100 KB on most consumer SKUs (though the RTX 5070 Ti laptop reports 128 KB), compared to 228 KB on SM100.

What SM120 does have
SM120 has fully functional FP4 Tensor Cores, accessible through warp-level mma.sync instructions inherited from the Ampere instruction path. The specific instruction for FP4 is mma.sync.aligned.kind::f8f6f4.m16n8k32, operating on a tile shape of 16×8×32.

This is a meaningful difference from the INT8 WMMA instructions used in the ParagEkbote/model-kernels kernel, which operate on a 16×16×16 tile shape on SM75/SM80. The FP4 instruction processes a K-dimension of 32 elements per cycle instead of 16, but produces a narrower output tile (8 columns instead of 16). This changes how tiles are partitioned, how accumulators are organized in registers, and how the inner loop over K interacts with the MMA calls.

Why this matters
This finding is poorly documented across the ecosystem. Most Blackwell Tensor Core tutorials and references target SM100 exclusively and either ignore SM120 or simply warn that it is "different" without explaining how. The practical consequence for any developer targeting consumer Blackwell GPUs for FP4 compute is clear: use mma.sync rather than tcgen05, manage accumulators in registers rather than TMEM, and design tile sizes around the tighter shared memory budget. This is the architectural foundation that the rest of this kernel is built on.
