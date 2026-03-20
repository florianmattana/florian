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

Between the two MMAs: softmax, quantization, and repacking

The fused attention kernel has three main steps: compute the scores (first MMA), apply softmax, then multiply by V (second MMA). The first and last steps are matrix multiplications handled by the Tensor Cores. What happens in between is the hard part, and it is where most of the engineering decisions live.

The problem

After the first MMA, we have the attention scores S sitting in FP32 registers across the 32 threads of each warp. These are raw dot products. We need to turn them into probabilities (softmax), then feed them into the second MMA as operand A. But the second MMA expects its input in FP4 E2M1, not FP32. So we need to do three things without leaving the registers: apply softmax, quantize from FP32 to FP4, and rearrange the data between threads because the output layout of the first MMA does not match the input layout of the second MMA.

Why fuse softmax and quantization

The naive approach would be to do softmax first (find the max, compute exponentials, sum, divide), and then in a second pass quantize the result to FP4 (find the absmax per block, divide, convert). That means reading every value from registers twice.

SageAttention3 fuses both passes into one. While scanning the registers to find the row max for softmax, they simultaneously compute the absmax per block of 16 elements for the FP4 scale factor. While applying the exponential, they fold the FP4 scaling directly into the math. One pass over the data, two results.

The trade-off: the code is significantly more complex. Every loop iteration does more work. But since registers are the most precious resource on SM120, reading them once instead of twice is worth the complexity.

The exp2 trick

Standard softmax computes exp(x - max). The GPU has no native hardware instruction for exp (base e). It does have a native instruction for exp2 (base 2), called ex2.approx.ftz.f32, which runs in a single cycle on the Special Function Unit.

Since exp(x) = 2^(x * log2(e)), we can replace every exp call with an exp2 call by pre-multiplying x by log2(e), which is approximately 1.4427. This constant can be folded into the softmax scale factor (1/sqrt(D)), so we pay for one multiplication instead of two. The compiler can then fuse the multiply and subtract into a single fma instruction.

Hypothesis: this optimization is well-established in GPU programming and is used by FlashAttention, SageAttention, and most production softmax implementations. We adopt it directly.

Two max operations, one pass

During the single pass over S, we compute two different max values.

The first is the row max for softmax. This is the maximum value across an entire row of S (for example 64 elements if BK=64). It requires a warp shuffle reduction across 4 threads (XOR with offsets 1 and 2), because 4 threads jointly hold the elements of one row.

The second is the absmax per block of 16 elements for the FP4 scale factor. This is a more local maximum. It only requires a warp shuffle across 2 threads (XOR with offset 1), because 2 neighboring threads hold one block of 16 elements.

Both are computed in the same loop. When a thread scans its registers to accumulate the row max, it simultaneously tracks the block absmax. No extra register reads.

Trade-off: the block absmax determines the FP4 scale factor. A block of 16 elements means one scale factor per 16 values (consistent with scale_vec::2X). Smaller blocks would give better precision but more scale factors to store and compute. Larger blocks would save overhead but lose precision on blocks with mixed magnitudes.

Quantizing P from FP32 to FP4

After softmax and scaling, each element of P has been divided by its block absmax, bringing values into the range that FP4 E2M1 can represent (0 to 6 for positive values after softmax). The actual conversion uses the PTX instruction cvt.rn.satfinite.e2m1x2.f32, which takes 2 FP32 values and outputs 2 FP4 values packed into a single byte. Four calls to this instruction fill one 32-bit register with 8 FP4 values.

This is a hardware-native conversion. The GPU handles rounding (round to nearest) and clamping (satfinite means values too large are clamped to the maximum representable value, not set to infinity). No manual bit manipulation is needed.

Trade-off: this is where precision loss happens. FP4 E2M1 has only 16 possible values. Softmax outputs are probabilities between 0 and 1, often with subtle differences (for example 0.12 vs 0.14). FP4 cannot distinguish these. SageAttention3 mitigates this by using per-block scaling, but some information is inevitably lost. The paper reports that for most practical models (diffusion, language), this loss does not significantly affect output quality.

Repacking: from accumulator layout to operand layout

The first MMA produces S in the accumulator layout (layout D). The second MMA expects P in the operand layout (layout A). These two layouts assign different matrix elements to different threads and different registers within each thread.

Think of it this way: 32 threads each hold a few cards with numbers on them. After the first MMA, each thread has specific cards determined by layout D. The second MMA says "I need each thread to hold different specific cards, determined by layout A." The values are the same, but they need to physically move between threads.

This rearrangement can be done via warp shuffles (threads exchange register values directly) or by writing to shared memory and reading back in the new order. SageAttention3 uses CuTe layout abstractions (LayoutP and LayoutSFP in kernel_traits.h) to handle this mapping. For our inline PTX kernel on SM120, we will need to implement this rearrangement manually.

Trade-off: warp shuffles are faster (register to register, no memory access) but limited to exchanging between threads within the same warp. Shared memory is more flexible but adds latency and requires synchronization. The choice depends on how different the two layouts are. If only a few elements need to move between threads, shuffles are sufficient. If the mapping is complex, shared memory might be simpler to implement correctly even if slower.

This repacking step is the single most error-prone part of the kernel. A mistake produces silent wrong results with no crash or warning. It is also the least documented part of existing implementations.

What SageAttention3 uses that we cannot

SageAttention3 targets SM100 (RTX 5090 and datacenter GPUs). It uses tcgen05.mma (fifth-generation Tensor Core instructions), TMA (Tensor Memory Access for hardware-managed loads), TMEM (Tensor Memory), and CuTe/CUTLASS abstractions. None of these are available on SM120 (RTX 5070 Ti, 5080).

What we keep from their approach: the fused softmax-plus-quantization logic, the exp2 trick, the dual max computation, and the cvt.rn.satfinite.e2m1x2.f32 instruction for FP4 conversion. These are all valid on SM120.

What we replace: TMA with cp.async, tcgen05.mma with mma.sync, CUTLASS pipeline abstractions with manual __syncthreads and cp.async.wait_group, and CuTe layout abstractions with explicit address calculations (as in gau-nernst's SM120 MXFP8 kernel).

Testing the FP4 MMA instruction on SM120

Before writing a full kernel, we needed to verify that a single FP4 MMA instruction works correctly on our GPU. This turned out to be much harder than expected, and the debugging process revealed critical undocumented details about how Blackwell handles FP4 data.

The minimal test kernel

The idea is simple: fill matrices A and B with known FP4 values, run one MMA instruction, and check whether the output matches the expected dot product. We launch a single warp (32 threads) and each thread declares its own registers for A, B, scale factors, and the accumulator D. After the MMA, each thread prints its four accumulator values.

The instruction we target is the block-scaled variant:

mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e2m1.f32.ue8m0
Each piece of this string matters. kind::mxf8f6f4 selects the instruction family that supports FP8, FP6, and FP4 types. block_scale enables hardware block scaling. scale_vec::1X means one scale factor per 32 elements along K. m16n8k32 defines the tile shape: 16 rows, 8 columns, 32 elements along the K dimension. e2m1.e2m1 specifies that both A and B are in FP4 E2M1 format. ue8m0 specifies the scale factor format.

First surprise: the tile shape

We initially assumed the FP4 instruction would use shape m16n8k64 since FP4 values are 4 bits and the hardware processes 32 bytes per step (32 bytes / 0.5 bytes per element = 64 elements). Every attempt to compile with m16n8k64 failed with "Incorrect instruction type specified for mma with shape m16n8k64".

After reading CUTLASS source code and NVIDIA forum posts, we discovered that kind::mxf8f6f4 always uses shape m16n8k32 regardless of the element type. The K dimension in the shape name refers to the number of 8-bit containers, not the number of logical elements. For FP8, 32 containers hold 32 elements. For FP4, 32 containers still hold 32 elements because each FP4 value is padded into an 8-bit container.

Second surprise: FP4 values are stored in 8-bit containers

This was the hardest bug to find. We encoded FP4 1.0 as nibble 0x2 and packed eight of them into a 32-bit register as 0x22222222. The MMA ran without errors but produced 2.0 instead of the expected 64.0. We tested multiple values and the results were internally consistent but wrong.

The answer came from a Discord discussion quoting the PTX documentation:

"When .kind is either of .kind::mxf8f6f4 or .kind::f8f6f4, the individual 4-bit floating point type elements must be packed in an 8-bit container. The matrix element of type .e2m1 resides in central 4 bits of the 8-bit container with padding in the upper 2 bits and lower 2 bits of the container."

Each FP4 value occupies one full byte. The 4-bit E2M1 value sits in bits 5 through 2, with two zero bits above and two zero bits below. So FP4 1.0 (0010 in E2M1) becomes 00 0010 00 in binary, which is 0x08, not 0x22.

With the correct encoding 0x08080808, the MMA produces 32.0, which is exactly right: 32 elements along K, each contributing 1.0 times 1.0.

Third surprise: scale_vec::2X is not available

We planned to use scale_vec::2X for finer quantization granularity (one scale factor per 16 elements instead of 32). The compiler rejected it with "Illegal modifier .scale_vec::2X for instruction mma". On SM120 with kind::mxf8f6f4, only scale_vec::1X is supported. The 2X option exists on SM100 with the kind::mxf4nvf4 instruction family, which packs FP4 values as true 4-bit nibbles and runs at twice the throughput.

What this means for the kernel

On SM120 with kind::mxf8f6f4, FP4 runs at half the theoretical throughput because each 4-bit value wastes 4 bits of padding. The alternative instruction family kind::mxf4nvf4 avoids this waste but is not available on SM120. This is a hardware limitation of consumer Blackwell GPUs that datacenter parts (SM100) do not have.

For our fused attention kernel, this means we work with scale_vec::1X (one scale per 32 elements) and accept the throughput penalty. The encoding rule (center the 4 bits in an 8-bit container) must be applied everywhere: when quantizing activations to FP4, when repacking P between the two MMAs, and when loading pre-quantized weights from memory.

## 4. Encoding FP32 to FP4 E2M1: Building the Quantization Function

### Why we need an encoding function

The MMA test validated that the hardware produces correct results when given properly encoded FP4 data. But in that test, we hardcoded the register values (`0x08080808`). In the real kernel, Q, K, and V arrive as FP16 or FP32 tensors from PyTorch. Every value must be converted to FP4 E2M1 at runtime, inside the kernel, without a separate quantization pass. This means we need a device function that takes a float and returns the corresponding 8-bit container.

### Anatomy of FP4 E2M1

The format has 4 bits: 1 sign bit (S), 2 exponent bits (E), and 1 mantissa bit (M). The exponent bias is 1, computed from the standard IEEE formula: bias = 2^(number of exponent bits - 1) - 1 = 2^1 - 1 = 1. This bias is a fixed property of the format, not a per-value parameter.

Decoding follows the same rules as IEEE 754. For normalized numbers (E > 0): value = (-1)^S × 1.M × 2^(E - bias). The `1.M` notation means an implicit leading 1 followed by the mantissa bit after a binary point. With M = 0, `1.M` = `1.0` in binary = 1.0 in decimal. With M = 1, `1.M` = `1.1` in binary = 1.5 in decimal (because the first digit after the binary point represents 2^(-1) = 0.5).

For denormalized numbers (E = 0): value = (-1)^S × 0.M × 2^(1 - bias). The implicit leading bit becomes 0 instead of 1. This allows representation of values smaller than the smallest normalized number.

The complete value table:

| Nibble | S | E  | M | Value |
|--------|---|----|---|-------|
| 0000   | 0 | 00 | 0 | +0.0  |
| 0001   | 0 | 00 | 1 | +0.5  |
| 0010   | 0 | 01 | 0 | +1.0  |
| 0011   | 0 | 01 | 1 | +1.5  |
| 0100   | 0 | 10 | 0 | +2.0  |
| 0101   | 0 | 10 | 1 | +3.0  |
| 0110   | 0 | 11 | 0 | +4.0  |
| 0111   | 0 | 11 | 1 | +6.0  |
| 1000   | 1 | 00 | 0 | -0.0  |
| 1001   | 1 | 00 | 1 | -0.5  |
| 1010   | 1 | 01 | 0 | -1.0  |
| 1011   | 1 | 01 | 1 | -1.5  |
| 1100   | 1 | 10 | 0 | -2.0  |
| 1101   | 1 | 10 | 1 | -3.0  |
| 1110   | 1 | 11 | 0 | -4.0  |
| 1111   | 1 | 11 | 1 | -6.0  |

Only 16 possible values. The distribution is non-uniform: resolution is finest near zero (steps of 0.5 between 0.0, 0.5, 1.0, 1.5) and coarsest at the extremes (a jump of 2.0 between 4.0 and 6.0). This is inherent to floating-point representation and is the fundamental tradeoff of aggressive quantization.

### Building the encoding function step by step

#### Step 1: Extract the sign

The FP4 value table is symmetric: the positive and negative halves are mirrors of each other, differing only in the sign bit. This means we can separate the problem into two independent parts: determine the sign, then find the correct magnitude. We extract the sign, work on the absolute value, and reattach the sign at the end.

```cuda
__device__ uint8_t encode_fp4_e2m1(float val) {
    uint8_t sign = (val < 0.f) ? 1 : 0;
    float abs_val = fabsf(val);
    return 0; // placeholder
}
Copy
fabsf is a CUDA intrinsic that returns the absolute value of a float. The ternary operator (condition) ? a : b evaluates to a if the condition is true, b otherwise. After these two lines, sign holds 0 or 1, and abs_val holds the positive magnitude. The float itself uses a completely different 32-bit format internally, so we cannot simply "read the first bit" as we would with a finished FP4 nibble—we need this explicit comparison.

Step 2: Find the nearest FP4 value by rounding
Given abs_val, we need to find which of the 8 positive FP4 values (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0) is closest. The standard approach is to compare against midpoints between consecutive values. The midpoint is where rounding switches from the lower value to the higher one:

Between	Midpoint
0.0 and 0.5	0.25
0.5 and 1.0	0.75
1.0 and 1.5	1.25
1.5 and 2.0	1.75
2.0 and 3.0	2.5
3.0 and 4.0	3.5
4.0 and 6.0	5.0
Values above 5.0 are clamped (saturated) to 6.0, the maximum representable magnitude. This is implemented as a chain of comparisons starting from the largest value:

uint8_t nibble;
if      (abs_val >= 5.0f)  nibble = 0x7;  // 6.0
else if (abs_val >= 3.5f)  nibble = 0x6;  // 4.0
else if (abs_val >= 2.5f)  nibble = 0x5;  // 3.0
else if (abs_val >= 1.75f) nibble = 0x4;  // 2.0
else if (abs_val >= 1.25f) nibble = 0x3;  // 1.5
else if (abs_val >= 0.75f) nibble = 0x2;  // 1.0
else if (abs_val >= 0.25f) nibble = 0x1;  // 0.5
else                       nibble = 0x0;  // 0.0
On a GPU, this chain of comparisons with compile-time constants is efficient: the compiler converts it to predicated instructions without branch divergence. A lookup table alternative would require a shared memory or constant memory access, which would be slower for such a short sequence.

The nibble values (0x0 through 0x7) are the 3-bit magnitude encodings from the FP4 table: exponent and mantissa without the sign bit.

Step 3: Attach the sign bit
The sign bit occupies position 3 of the nibble (the most significant bit of the 4-bit FP4 value). We count bit positions from right to left starting at 0:

position:  3  2  1  0
           S  E  E  M
To place the sign at position 3, we shift it left by 3 and combine it with the magnitude using a bitwise OR:

nibble |= (sign << 3);
The operator << shifts bits to the left. If sign = 1, then 1 << 3 produces binary 1000. The operator | (bitwise OR) combines two bit patterns: wherever either input has a 1, the result has a 1. Since the sign bit (position 3) and the magnitude bits (positions 0–2) occupy non-overlapping positions, the OR assembles them without interference.

The shorthand nibble |= x is equivalent to nibble = nibble | x.

Example: encoding -1.0. The magnitude nibble is 0010 (+1.0). The sign is 1. After 1 << 3 = 1000, the OR gives 0010 | 1000 = 1010, which is the correct FP4 encoding for -1.0.

Step 4: Place in the 8-bit container
As discovered during MMA testing, kind::mxf8f6f4 on SM120 requires each FP4 value to sit at bits 5 through 2 of an 8-bit container, with bits 7–6 and bits 1–0 set to zero. This is achieved with a left shift of 2:

return (uint8_t)(nibble << 2);
The final byte layout:

bit:     7  6  5  4  3  2  1  0
         0  0  S  E  E  M  0  0
Example: nibble 1010 (-1.0) shifted left by 2 becomes 00101000 = 0x28.

The complete function
__device__ uint8_t encode_fp4_e2m1(float val) {
    uint8_t sign = (val < 0.f) ? 1 : 0;
    float abs_val = fabsf(val);

    uint8_t nibble;
    if      (abs_val >= 5.0f)  nibble = 0x7;
    else if (abs_val >= 3.5f)  nibble = 0x6;
    else if (abs_val >= 2.5f)  nibble = 0x5;
    else if (abs_val >= 1.75f) nibble = 0x4;
    else if (abs_val >= 1.25f) nibble = 0x3;
    else if (abs_val >= 0.75f) nibble = 0x2;
    else if (abs_val >= 0.25f) nibble = 0x1;
    else                       nibble = 0x0;

    nibble |= (sign << 3);

    return (uint8_t)(nibble << 2);
}
Validation
We tested the encoding function against known values, then fed the encoded values into the MMA instruction to verify end-to-end correctness.

Unit tests of the encoding function:

Input	Expected nibble	Expected container	Observed
1.0	0010	0x08	0x08 ✓
-1.0	1010	0x28	0x28 ✓
6.0	0111	0x1C	0x1C ✓
1.2	0010 (rounded to 1.0)	0x08	0x08 ✓
The rounding test (1.2 → 1.0) confirms that the midpoint logic works: 1.2 < 1.25 (the midpoint between 1.0 and 1.5), so the function correctly rounds down.

Packing encoded values into 32-bit registers
The MMA instruction expects its operands as 32-bit registers. Each register holds 4 encoded FP4 values (one per byte). Packing uses bitwise shifts and OR to place each byte at the correct position within the 32-bit word:

uint8_t e0 = encode_fp4_e2m1(1.0f);
uint32_t packed = e0 | (e0 << 8) | (e0 << 16) | (e0 << 24);
The shift amounts correspond to byte boundaries within the 32-bit register: e0 occupies bits 7–0, e0 << 8 occupies bits 15–8, e0 << 16 occupies bits 23–16, and e0 << 24 occupies bits 31–24. Since these ranges do not overlap, the bitwise OR assembles all four bytes into a single word. The resulting value is 0x08080808, identical to what we hardcoded in the earlier MMA test.

End-to-end MMA validation
With the encoded and packed values replacing the hardcoded constants:

uint8_t e0 = encode_fp4_e2m1(1.0f);
uint32_t packed = e0 | (e0 << 8) | (e0 << 16) | (e0 << 24);

uint32_t A[4] = {packed, packed, packed, packed};
uint32_t B[2] = {packed, packed};
The MMA produced 32.0 across all 32 lanes, matching the expected result (32 K-elements, each 1.0 × 1.0, summed).

We repeated the test with encode_fp4_e2m1(2.0f), which produced 128.0 (= 32 × 2.0 × 2.0), confirming that the full pipeline—encoding, packing, and hardware MMA—works correctly for non-trivial values.

The encoding function is now validated and ready to be used in the fused attention kernel. It lives in common.h so that both test files and the final kernel can include it.
