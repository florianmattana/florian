---
title: Building an FP4 Fused Attention Kernel for Consumer Blackwell GPUs (SM120)
description: Design document for a fused attention kernel targeting consumer Blackwell GPUs
header: FP4 Fused Attention Kernel for SM120
---

## 1. Problem Statement

The efficiency of attention computation is critical for ML inference on consumer hardware, given that its computational cost grows quadratically with sequence length: doubling the number of tokens quadruples the size of the attention score matrix and the number of operations required.

Recent work has demonstrated that FP4 Tensor Cores on Blackwell GPUs can deliver up to 5x speedup over FP16 attention through quantized fused kernels. However, existing implementations target a narrow hardware slice. SageAttention3 (Tsinghua, NeurIPS 2025) achieves 1038 TOPS but has been developed and validated exclusively on the RTX 5090 (GB202, 170 SMs). FlashAttention-4 targets Blackwell datacenter GPUs (SM100). The broader SM120 consumer lineup (RTX 5080, RTX 5070 Ti, RTX 5070, RTX 5060 Ti) remains without a functional FP4 fused attention kernel. Community reports confirm that SageAttention3 crashes on the RTX 5070 Ti and fails to compile on the RTX 5060 Ti, leaving these GPUs unable to exploit their native FP4 Tensor Core capabilities.

This represents a significant missed opportunity. On a RTX 5070 Ti (GB205, 46 SMs), the FP4 Tensor Cores deliver approximately 474 dense TFLOPS, a 4x throughput advantage over FP16 Tensor (123.5 TFLOPS) and 2x over INT8 Tensor (246.9 TFLOPS). Simultaneously, the 4-bit data format provides a 4x reduction in memory traffic compared to FP16, directly alleviating the primary bottleneck on memory-bandwidth-constrained consumer GPUs (672 GB/s on RTX 5070 Ti versus 1,792 GB/s on RTX 5090). The alternative, non-fused FP4 GEMM kernels such as VincentKaufmann's fp4-cuda-kernel (143 TFLOPS on SM120), requires materializing the full N×N attention score matrix in VRAM between the QKᵀ and softmax·V stages, which is prohibitive for long sequences on 12 GB devices.

I design and implement an FP4 fused attention kernel specifically targeting the SM120 consumer GPU family beyond the RTX 5090. The kernel computes softmax(QKᵀ/√d)·V in a single pass using FP4 E2M1 quantization with hardware-native block scaling (one UE8M0 scale factor per 16 elements), online softmax for numerical stability, and tiled shared memory management optimized for the 128 KB per-SM shared memory and 12 GB VRAM constraints of consumer Blackwell GPUs.

## 2. Design Decision: Instruction Path for FP4 Tensor Cores

### The three options

Before writing any kernel code, the first architectural decision was to determine how to invoke the FP4 Tensor Cores. Three options were evaluated.

Option A consists of writing the matrix multiplications in inline MMA PTX, providing full control over register placement, shared memory access patterns, and the compute pipeline. The cost is complexity: there is no C++ abstraction layer, and the developer is responsible for encoding operands into the correct register layout expected by the hardware.

Option B consists of using CuTe (the layout and data movement sublibrary within CUTLASS 3.x) to handle operand loading and memory layouts, while writing the fused attention loop and online softmax manually. This reduces the burden of register management but introduces a dependency on CUTLASS internals, and the interaction between CuTe's tile abstractions and a custom fused loop with softmax in the middle is not straightforward.

Option C consists of starting from an existing fused INT8 attention kernel (specifically, the ParagEkbote/model-kernels kernel that I have previously contributed to) and surgically replacing the WMMA INT8 matrix multiplications with FP4 equivalents. This minimizes risk since the tiling, online softmax, and memory management are already proven, but it limits the scope of the experience to a surface-level swap without deep knowledge of what the hardware is actually doing.

### Why Option A

The core constraint of a fused attention kernel is that the intermediate attention score matrix from QKᵀ must never be materialized in VRAM. It must stay in shared memory and registers between the first matrix multiplication (QKᵀ), the online softmax, and the second matrix multiplication (scores·V). Any approach that delegates the matrix multiplications to an external library risks breaking this fusion by requiring data to exit and re-enter the kernel's local memory hierarchy.

Option B was considered but set aside: CuTe is designed around standalone GEMM operations, and inserting a custom softmax between two CuTe-managed matrix multiplications would require fighting the abstraction rather than leveraging it.

Option C was rejected on principle. The goal of this project is not to produce a working kernel by the shortest path, but to share the full stack from PTX instructions to fused attention, and to document the difficulties encountered along the way. Starting from a working kernel and patching it would shortcut that journey.

Option A was selected: CUDA C++ for the kernel structure (launch configuration, shared memory allocation, tiling loop, online softmax, output writeback) with inline PTX assembly specifically for the Tensor Core matrix multiply instructions.

### The first question: which PTX instruction activates FP4 Tensor Cores?

With Option A selected, the immediate problem was to identify which PTX instruction to use. The working hypothesis was that all Blackwell GPUs share the same Tensor Core instruction set. The most documented path for FP4 on Blackwell is tcgen05.mma, a 5th generation Tensor Core instruction that operates on Tensor Memory (TMEM), supports massive MMA shapes up to m128×n256×k16, and is the foundation of high-performance datacenter kernels like SageAttention3 and DeepGEMM.

This hypothesis turned out to be wrong.

### SM100 vs SM120: a critical architectural split

SM100 (datacenter Blackwell: B200, B300) and SM120 (consumer Blackwell: RTX 5070, 5070 Ti, 5080, 5090) do not share the same instruction set for Tensor Core operations. Three independent sources confirm this.

CUTLASS issue #2800 states explicitly: "FP4 on SM120/SM121 is usable through Ampere-style mma instructions rather than tcgen05." The NVIDIA developer forum post on FP4 performance on DGX Spark identifies the specific instructions: "OMMA and QMMA are warp-level MMA instructions that work on SM120/SM121. These are the tensor core instructions that CUTLASS accesses through CuTe." CUTLASS issue #3044 reveals the exact PTX syntax: mma.sync.aligned.kind::f8f6f4.m16n8k32 when targeting sm_120a.

### What SM120 does not have

SM120 lacks several features that SM100 relies on for peak FP4 performance. There is no tcgen05.mma instruction. There is no Tensor Memory (TMEM), the dedicated 128×512 accumulator memory that SM100 uses to reduce register pressure. There is no multicast and no CTA pairs, meaning no cooperative 2-SM execution. The cluster size is fixed at 1×1×1. Shared memory per SM is approximately 100 KB on most consumer SKUs (though the RTX 5070 Ti laptop reports 128 KB), compared to 228 KB on SM100.

### What SM120 does have

SM120 has fully functional FP4 Tensor Cores, accessible through warp-level mma.sync instructions inherited from the Ampere instruction path. The specific instruction for FP4 is mma.sync.aligned.kind::f8f6f4.m16n8k32, operating on a tile shape of 16×8×32.

This is a meaningful difference from the INT8 WMMA instructions used in the ParagEkbote/model-kernels kernel, which operate on a 16×16×16 tile shape on SM75/SM80. The FP4 instruction processes a K-dimension of 32 elements per cycle instead of 16, but produces a narrower output tile (8 columns instead of 16). This changes how tiles are partitioned, how accumulators are organized in registers, and how the inner loop over K interacts with the MMA calls.

### Why this matters

This finding is poorly documented across the ecosystem. Most Blackwell Tensor Core tutorials and references target SM100 exclusively and either ignore SM120 or simply warn that it is "different" without explaining how. The practical consequence for any developer targeting consumer Blackwell GPUs for FP4 compute is clear: use mma.sync rather than tcgen05, manage accumulators in registers rather than TMEM, and design tile sizes around the tighter shared memory budget. This is the architectural foundation that the rest of this kernel is built on.

## 3. Instruction Selection and Block Scaling Tradeoffs

### Hardware-native block scaling

A key concern when designing an FP4 kernel is whether the block scaling must be implemented manually in the kernel, or whether the hardware handles it natively. The PTX documentation confirms that the mma.sync instruction with the following .kind qualifiers performs matrix multiplication with built-in block scaling:

- .kind::mxf8f6f4
- .kind::mxf4
- .kind::mxf4nvf4

The operation takes the form: D = (A × scale_A) × (B × scale_B) + C. Both operands A and B carry independent scale factors because they originate from different tensors (for instance, Q and K in the attention computation) with different value distributions. The hardware applies the dequantization during the multiply-accumulate, with no additional instructions required from the kernel. This eliminates an entire class of complexity that would otherwise consume shared memory bandwidth and register space.

### Register budget per MMA instruction

Before selecting the specific instruction variant, I mapped the register footprint of a single FP4 MMA instruction with shape m16n8k32 to identify potential register pressure issues.

The matrix A has shape 16×32 = 512 FP4 elements. At 4 bits per element, this is 2048 bits, fitting in 64 registers of 32 bits, distributed across 32 threads in a warp: 2 registers per thread. The matrix B has shape 32×8 = 256 FP4 elements, requiring 1 register per thread. The accumulator D has shape 16×8 = 128 elements stored in FP32 (32 bits each), requiring 4 registers per thread. The total is 7 registers per thread for a single MMA instruction.

For comparison, the INT8 WMMA m16n16k16 instruction used in prior fused attention kernels produces a 16×16 output in INT32, requiring 256 elements of 32-bit accumulation distributed across the warp, resulting in a heavier register footprint. The FP4 instruction benefits from doubly compact operands (4 bits vs 8 bits) and a narrower output tile (16×8 vs 16×16). This reduced register pressure leaves more headroom for the online softmax state (row_max, row_sum) and for double-buffering tiles in registers, both of which are critical in a fused attention kernel.

A practical consequence of the 16×8 output shape is that covering a 16×16 result tile requires two MMA instructions side by side, one for columns 0 through 7 and one for columns 8 through 15.

### Choosing between instruction variants

The three .kind qualifiers supporting FP4 E2M1 differ in their block scaling granularity:

.kind::mxf8f6f4 with E2M1 uses scale_vec::1X, providing 1 scale factor per group of 32 elements. .kind::mxf4 with E2M1 uses scale_vec::2X, providing 2 scale factors per group, equivalent to 1 scale factor per 16 elements. .kind::mxf4nvf4 with E2M1 supports scale_vec::2X or scale_vec::4X, providing 2 or 4 scale factors per group, with an additional option for ue4m3 scale format.

This is a precision-versus-overhead tradeoff that can be evaluated quantitatively.

### Quantifying the tradeoff

For a representative 64×64 tile containing 4096 FP4 elements, the data itself occupies 4096 × 0.5 bytes = 2048 bytes. The scale factor overhead depends on the granularity. With scale_vec::1X, there are 4096 / 32 = 128 scale factors at 1 byte each (UE8M0 format), adding 128 bytes for a 6.25% overhead. With scale_vec::2X, there are 4096 / 16 = 256 scale factors, adding 256 bytes for a 12.5% overhead.

On the register side, the scale factors are encoded in UE8M0 (8 bits each). Moving from 1 byte to 2 bytes of scale data per operand still fits within a single 32-bit register. There is zero additional register cost.

The precision benefit of finer-grained scaling is particularly relevant for the attention score matrix P, whose values lie in [0, 1] after softmax. SageAttention3 identifies this as a core challenge (C2 in their paper): when small post-softmax values are quantized to FP4, the scale factors collapse into an extremely narrow dynamic range, causing significant accuracy loss. Doubling the scale factor granularity from 1-per-32 to 1-per-16 helps contain this effect by allowing each smaller group to have its own adapted scale, at a cost of only 6.25 additional percentage points of memory overhead and zero additional register pressure.

### Final instruction choice

Based on this analysis, the selected instruction is:

mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k32

With E2M1 data format, UE8M0 scale format, and a tile shape of 16×8×32. This provides the best balance between quantization precision and resource overhead for a fused attention kernel targeting SM120 consumer GPUs.

## 4. Tiling Strategy: Shared Memory vs Register Pressure

### Understanding the two GEMMs in attention

The attention computation consists of two matrix multiplications with a softmax in between. For a sequence of N tokens with head dimension D:

GEMM 1 computes the attention scores: Q(N,D) × Kᵀ(D,N) = S(N,N). Each element S[i,j] is the dot product between token i's query vector and token j's key vector, measuring how much token i should attend to token j. The transposition of K is necessary because both Q and K store tokens as rows, and the dot product requires matching the row of Q with the row of K, which becomes a column after transposition.

GEMM 2 computes the output: P(N,N) × V(N,D) = O(N,D), where P is the softmax of S. The result O has the same shape as Q: each token gets back a D-dimensional vector, now enriched with information from other tokens weighted by the attention scores.

The critical problem is that S has shape N×N. For N=4096 tokens in FP32, S would require 4096×4096×4 bytes = 64 MB. The entire purpose of a fused kernel is to never materialize this matrix in VRAM.

### The fused attention structure

The kernel avoids materializing S by fixing a tile of Q in shared memory and streaming tiles of K and V sequentially through the same shared memory space. For each tile of K, the kernel computes the partial attention scores Q·Kᵀ, applies online softmax to maintain numerical stability, then immediately loads the corresponding tile of V into the same shared memory slot to compute the weighted output scores·V. The attention scores never leave the registers.

This creates two levels of looping. The outer loop iterates over tiles of K/V (different groups of tokens). The inner loop iterates over chunks of the D dimension to accumulate the dot products. Each dot product is built progressively: with D=128 and a chunk size of 32 (imposed by the MMA instruction), the inner loop runs 4 iterations, each contributing a partial sum to the same accumulator registers.

### The tiling tradeoff

The tile sizes BQ (rows of Q) and BK (rows of K/V) determine both performance and resource consumption. Two constraints pull in opposite directions.

Increasing BQ improves data reuse. Each tile of K loaded from VRAM is multiplied against more rows of Q, increasing the ratio of compute to memory access. This pushes the kernel toward being compute-bound rather than memory-bound, which is desirable on a GPU with 474 TFLOPS of FP4 compute but only 672 GB/s of memory bandwidth.

However, increasing BQ or BK increases the accumulator size. The accumulator for the block S(BQ, BK) contains BQ×BK elements in FP32 (32 bits each), stored in registers across the threads of the block. Registers are a fixed resource: 65536 per SM on the RTX 5070 Ti.

### Why FP4 is register-bound, not shared-memory-bound

This is a key insight of the design. With FP4 data at 4 bits per element plus UE8M0 scale factors at 8 bits per 16 elements, the shared memory footprint is extremely compact. A tile of Q(64,128) requires only 4608 bytes (4096 bytes of data + 512 bytes of scale factors). A tile of K or V requires the same. Since Q and K/V coexist in shared memory while K and V alternate in the same slot, the total shared memory usage is approximately 9 KB out of 128 KB available. That is only 7% utilization.

The bottleneck is the accumulator. The attention score block S(BQ, BK) is stored in FP32 regardless of the operand precision, because accumulating in FP4 would destroy the result through rounding errors. This creates an asymmetry: operands are 4 bits wide, but the accumulator is 32 bits wide, an 8× ratio.

### Dimensioning: BQ=64, BK=64

With BQ=64 and BK=64, the accumulator contains 64×64 = 4096 FP32 elements. Distributed across 4 warps (128 threads), each thread holds 4096/128 = 32 accumulator registers. Adding registers for MMA operands (~8), softmax state (row_max, row_sum, correction), loop counters, and addresses, the total is approximately 50 registers per thread. This allows 65536/50 ≈ 1310 threads per SM, which is 85% of the maximum 1536 threads. This is healthy occupancy.

For comparison, BQ=128 and BK=128 would require 128×128 = 16384 accumulator elements, giving 128 registers per thread for the accumulator alone, approximately 158 total. This limits occupancy to 414 threads per SM (27%), which significantly reduces the ability to hide memory latency.

The shared memory cost at BQ=64, BK=64 is only 9216 bytes, leaving over 118 KB unused. This confirms that registers, not shared memory, are the binding constraint for FP4 fused attention on SM120.

### MMA instruction mapping

With the selected MMA instruction shape of m16n8k32, covering the S(64,64) result block requires 4 MMA blocks along the rows (64/16=4) and 8 MMA blocks along the columns (64/8=8), for a total of 32 MMA instructions per accumulation iteration. With D=128 requiring 4 accumulation iterations (128/32=4), each tile of K consumes 32×4 = 128 MMA instructions to produce the complete partial attention scores.
