---
title: Building an FP4 Fused Attention Kernel for Consumer Blackwell GPUs (SM120)
description: Design document for a fused attention kernel targeting consumer Blackwell GPUs
header: FP4 Fused Attention Kernel for SM120
---
{% raw %}

## 1. Why This Kernel Needs to Exist

Attention is the bottleneck. Its cost grows quadratically with sequence length — double the tokens, quadruple the score matrix, quadruple the compute. On consumer GPUs with 12 GB of VRAM and 672 GB/s of bandwidth, this is where inference hits a wall.

Blackwell brought FP4 Tensor Cores to consumer hardware. On my RTX 5070 Ti (GB205, 46 SMs), the numbers are compelling: approximately 474 dense TFLOPS in FP4, compared to 123.5 TFLOPS in FP16 and 246.9 TFLOPS in INT8. That is a 4x throughput advantage over FP16, and the 4-bit data format simultaneously cuts memory traffic by 4x — directly attacking the bandwidth bottleneck that dominates attention on these GPUs.

The problem is that no one has written the kernel.

SageAttention3 (Tsinghua, NeurIPS 2025) achieves 1038 TOPS on the RTX 5090 (GB202, 170 SMs), but it targets SM100 hardware features that do not exist on the broader consumer lineup. Community reports confirm it crashes on the RTX 5070 Ti and fails to compile on the RTX 5060 Ti. FlashAttention-4 targets datacenter Blackwell exclusively. The RTX 5080, 5070 Ti, 5070, and 5060 Ti — the GPUs that most developers actually own — have fully functional FP4 Tensor Cores that no existing fused attention kernel can use.

The alternative is non-fused FP4 GEMM. [VincentKaufmann's fp4-cuda-kernel](https://github.com/VincentKaufmann/fp4-cuda-kernel) reaches 143 TFLOPS on SM120, which is respectable, but it requires materializing the full N×N attention score matrix in VRAM between the Q·Kᵀ and softmax·V stages. For a 4096-token sequence in FP32, that matrix alone is 64 MB. On a 12 GB device running multiple layers, this is prohibitive.

So the goal is clear: a fused attention kernel — softmax(Q·Kᵀ/√d)·V in a single pass — using FP4 E2M1 with hardware block scaling, targeting SM120 specifically. The attention scores stay in registers, never touch VRAM, and the entire computation runs on the Tensor Cores that the hardware provides but no software currently exploits.

This post documents the full build, including the parts where I got things wrong.

## 2. Choosing How to Talk to the Tensor Cores

### Three options

Before writing a single line of kernel code, I had to decide how to invoke the FP4 Tensor Cores. There were three realistic paths.

**Option A: inline PTX assembly.** Write the MMA instructions directly in `asm volatile` blocks inside a CUDA C++ kernel. This gives full control over register placement, shared memory access patterns, and the compute pipeline. The cost is that there is no abstraction layer — I am responsible for encoding operands into the exact register layout the hardware expects, and every mistake is silent.

**Option B: CuTe (CUTLASS 3.x).** Use NVIDIA's layout and data movement sublibrary to handle operand loading and register management, while writing the fused attention loop and online softmax manually. This reduces the register bookkeeping burden, but CuTe is designed around standalone GEMM operations. Inserting a custom softmax between two CuTe-managed matrix multiplications means fighting the abstraction rather than leveraging it — a problem the CUTLASS team themselves acknowledge is non-trivial.

**Option C: patch an existing kernel.** Start from the fused INT8 attention kernel in [ParagEkbote/model-kernels](https://github.com/ParagEkbote/model-kernels) (which I have previously contributed to) and surgically replace the WMMA INT8 matrix multiplications with FP4 equivalents. The tiling, online softmax, and memory management are already proven. The risk is low. But the learning is also low — it is a surface-level swap that does not require understanding what the hardware is actually doing.

### Why Option A

The core constraint of a fused attention kernel is that the intermediate score matrix S = Q·Kᵀ must never be materialized in VRAM. It must live in registers between the first MMA (Q·Kᵀ), the softmax, and the second MMA (P·V). Any approach that delegates the matrix multiplications to a library risks breaking this fusion by requiring data to exit and re-enter the kernel's local memory.

Option B was tempting, but after reading the CuTe documentation and several CUTLASS examples, I concluded that inserting a fused softmax-plus-quantization pass between two CuTe-managed GEMMs would require more effort than writing the MMA calls directly. CuTe wants to own the data flow, and I need to interrupt it.

Option C was rejected on principle. The goal of this project is not to produce a working kernel by the shortest path. It is to understand and document the full stack — from PTX instructions to fused attention — and to share the difficulties encountered along the way. Starting from a working kernel and patching it would shortcut that journey.

So: Option A. CUDA C++ for the kernel structure (launch config, shared memory, tiling, softmax, writeback), inline PTX for the Tensor Core instructions.

### First question: which PTX instruction?

With the approach decided, the immediate problem was identifying the correct PTX instruction. My working assumption was that all Blackwell GPUs share the same Tensor Core instruction set. The most documented path for FP4 on Blackwell is `tcgen05.mma` — a fifth-generation Tensor Core instruction that operates on Tensor Memory (TMEM), supports massive tile shapes up to m128×n256×k16, and is the foundation of SageAttention3 and DeepGEMM.

This assumption turned out to be wrong, and it cost me a few days.

### The SM100 vs SM120 split

SM100 (datacenter Blackwell: B200, B300) and SM120 (consumer Blackwell: RTX 5070 through 5090) do not share the same instruction set for Tensor Core operations. I pieced this together from three independent sources:

[CUTLASS issue #2800](https://github.com/NVIDIA/cutlass/issues/2800) states it explicitly: *"FP4 on SM120/SM121 is usable through Ampere-style mma instructions rather than tcgen05."*

An [NVIDIA developer forum post](https://forums.developer.nvidia.com/t/run-ptx-mma-sync-aligned-kind-mxf8f6f4-block-scale-scale-vec-1x-m16n8k32-on-sm-120a/329702) on FP4 performance identifies the specific instructions: *"OMMA and QMMA are warp-level MMA instructions that work on SM120/SM121."*

[CUTLASS issue #3044](https://github.com/NVIDIA/cutlass/issues/3044) reveals the exact PTX syntax: `mma.sync.aligned.kind::f8f6f4.m16n8k32` when targeting `sm_120a`.

What SM120 does *not* have: no `tcgen05.mma`, no Tensor Memory (TMEM) — the dedicated 128×512 accumulator memory that SM100 uses to reduce register pressure — no multicast, no CTA pairs, no cooperative 2-SM execution. The cluster size is fixed at 1×1×1. Shared memory per SM is approximately 128 KB on the RTX 5070 Ti, compared to 228 KB on SM100.

What SM120 *does* have: fully functional FP4 Tensor Cores, accessible through warp-level `mma.sync` instructions inherited from the Ampere instruction path, operating on a tile shape of 16×8×32.

This finding is poorly documented across the ecosystem. Most Blackwell Tensor Core tutorials target SM100 exclusively and either ignore SM120 or warn that it is "different" without explaining how. The practical consequence is clear: if you are targeting consumer Blackwell for FP4 compute, use `mma.sync`, manage accumulators in registers (not TMEM), and design tile sizes around the tighter resource budget. Everything that follows in this post is built on this foundation.

## 3. Picking the Right Instruction Variant

### Hardware-native block scaling

A key question when designing an FP4 kernel is whether block scaling — the per-group scale factor that maps low-precision values back to their original magnitude — must be implemented manually, or whether the hardware handles it. The answer determines an entire layer of complexity.

The PTX documentation confirms that `mma.sync` with certain `.kind` qualifiers performs block-scaled matrix multiplication natively. The operation is D = (A × scale_A) × (B × scale_B) + C. Both operands carry independent scale factors because they originate from different tensors (Q and K, for instance) with different value distributions. The hardware applies the dequantization during the multiply-accumulate. No additional instructions from the kernel. This eliminates what would otherwise be a significant source of shared memory bandwidth consumption and register pressure.

### Register budget

Before choosing between instruction variants, I mapped the register footprint of a single FP4 MMA instruction with shape m16n8k32.

Matrix A is 16×32 = 512 FP4 elements. At 4 bits each, that is 2048 bits, fitting in 64 registers of 32 bits, distributed across 32 threads: **2 registers per thread**. Matrix B is 32×8 = 256 FP4 elements: **1 register per thread**. The accumulator D is 16×8 = 128 FP32 elements: **4 registers per thread**. Total: **7 registers per thread** for a single MMA call.

For comparison, the INT8 WMMA m16n16k16 instruction I used in prior fused attention work produces a 16×16 output in INT32, with a heavier register footprint. The FP4 instruction benefits from doubly compact operands (4 bits vs 8 bits) and a narrower output tile (16×8 vs 16×16). This matters because every register saved is a register available for softmax state, loop counters, or double-buffering — all of which are critical in a fused kernel.

One practical consequence of the 16×8 output: covering a 16×16 result tile requires two MMA instructions side by side, one for columns 0–7 and one for columns 8–15.

### Three instruction families, three scaling granularities

The PTX ISA offers three `.kind` qualifiers that support FP4 E2M1:

**`kind::mxf8f6f4`** with E2M1 uses `scale_vec::1X` — one scale factor per 32 elements along K.

**`kind::mxf4`** with E2M1 uses `scale_vec::2X` — two scale factors per group, equivalent to one scale factor per 16 elements.

**`kind::mxf4nvf4`** with E2M1 supports `scale_vec::2X` or `4X`, with an additional option for UE4M3 scale format.

This is a precision-versus-overhead tradeoff, and I wanted to quantify it before committing.

### Quantifying the tradeoff

For a 64×64 tile of 4096 FP4 elements, the data itself occupies 4096 × 0.5 = 2048 bytes. With `scale_vec::1X`, there are 4096/32 = 128 scale factors at 1 byte each (UE8M0), adding 128 bytes — a 6.25% overhead. With `scale_vec::2X`, there are 4096/16 = 256 scale factors, adding 256 bytes — 12.5% overhead.

On the register side, moving from 1 to 2 bytes of scale data per operand still fits within a single 32-bit register. Zero additional register cost.

The precision benefit matters most for the attention score matrix P. After softmax, values lie in [0, 1], often with subtle differences (0.12 vs 0.14). SageAttention3 identifies this as a core challenge: when small post-softmax values are quantized to FP4, the scale factors collapse into an extremely narrow dynamic range. Doubling the granularity from 1-per-32 to 1-per-16 helps contain this effect, at a cost of 6.25 additional percentage points of memory overhead and zero register penalty.

Based on this analysis, I chose `kind::mxf4.block_scale.scale_vec::2X.m16n8k32` — the variant with finer scaling granularity.

### Third surprise: `scale_vec::2X` does not compile on SM120

This is where the plan met reality. The compiler rejected the instruction with: *"Illegal modifier .scale_vec::2X for instruction mma"*.

After further investigation, I found that on SM120 with `kind::mxf8f6f4`, only `scale_vec::1X` is supported. The `2X` option exists on SM100 with `kind::mxf4nvf4`, which packs FP4 values as true 4-bit nibbles (two per byte) and runs at twice the throughput.

On SM120 with `kind::mxf8f6f4`, FP4 runs at half the theoretical throughput because each 4-bit value is padded into an 8-bit container — 4 bits of data, 4 bits of waste. The alternative family `kind::mxf4nvf4` avoids this waste but is not available on SM120. This is a hardware limitation that datacenter Blackwell (SM100) does not have.

### The actual instruction

So the final instruction, after the failed `2X` attempt, is:

Copy
mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e2m1.f32.ue8m0


One scale factor per 32 elements. 8-bit containers for 4-bit values. Not what I originally wanted, but what the hardware actually supports.

### The two GEMMs in attention

The attention computation is two matrix multiplications with a softmax in between.

**GEMM 1** computes the attention scores: Q(N,D) × Kᵀ(D,N) = S(N,N). Each element S[i,j] is the dot product between token i's query vector and token j's key vector — measuring how much token i should attend to token j. K is transposed because both Q and K store tokens as rows, and the dot product requires matching Q's row with K's row, which becomes a column after transposition.

**GEMM 2** computes the output: P(N,N) × V(N,D) = O(N,D), where P = softmax(S). Each output token gets back a D-dimensional vector, now enriched with information from other tokens weighted by the attention scores.

The critical problem: S has shape N×N. For N=4096 tokens in FP32, that is 64 MB. The entire purpose of a fused kernel is to never materialize this matrix.

### How the fused kernel avoids materialization

The kernel fixes a tile of Q in shared memory and streams tiles of K and V through the same shared memory slot. For each tile of K: compute Q·Kᵀ (partial attention scores), apply online softmax in registers, immediately load the corresponding V tile into the same shared memory, compute P·V. The scores never leave the registers.

This creates two levels of looping. The outer loop iterates over tiles of K/V (different groups of tokens). The inner loop iterates over chunks of the D dimension to accumulate dot products. With D=128 and a chunk size of 32 (imposed by the MMA instruction), the inner loop runs 4 iterations per tile.

### Tile size: the binding constraint is registers, not shared memory

This was one of the more surprising findings of the design phase.

With FP4 data at 4 bits per element plus UE8M0 scale factors at 8 bits per 32 elements, the shared memory footprint is remarkably compact. A tile of Q(64,128) requires only 4608 bytes. A tile of K or V requires the same. Since Q stays resident while K and V alternate, total shared memory usage is approximately 9 KB out of 128 KB available — 7% utilization.

The bottleneck is the accumulator. The attention score block S(BQ, BK) is stored in FP32 regardless of operand precision, because accumulating in FP4 would destroy the result through rounding. This creates an asymmetry: operands are 4 bits wide, but the accumulator is 32 bits wide. An 8× ratio.

With BQ=64, BK=64: the accumulator contains 4096 FP32 elements. Distributed across 4 warps (128 threads), each thread holds 32 accumulator registers. Adding MMA operands (~8), softmax state (row_max, row_sum, correction factor), loop counters, and addresses: approximately 50 registers per thread. This allows 65536/50 ≈ 1310 threads per SM — 85% of the maximum 1536. Healthy occupancy.

For comparison, BQ=128, BK=128 would require 128 registers per thread for the accumulator alone (~158 total), limiting occupancy to 27%. At that point, the hardware cannot hide memory latency.

The shared memory cost at BQ=64, BK=64 is 9216 bytes, leaving 118 KB unused. Registers, not shared memory, are the binding constraint for FP4 fused attention on SM120.

### MMA mapping

With shape m16n8k32, covering S(64,64) requires 64/16 = 4 blocks along rows and 64/8 = 8 blocks along columns: 32 MMA instructions per accumulation step. With D=128 requiring 4 steps (128/32), each tile of K consumes 128 MMA instructions.n include it.

## 4. Testing the MMA Instruction (and Everything That Went Wrong)

### The plan

Before writing a full kernel with tiling, softmax, and shared memory management, I needed to verify that a single FP4 MMA instruction actually works on my GPU. The idea is simple: fill matrices A and B with known FP4 values, run one MMA, check that the output matches the expected dot product. If this does not work, nothing else matters.

I wrote a minimal test kernel: launch a single warp (32 threads), each thread declares its own registers for A, B, scale factors, and the accumulator D. After the MMA, each thread prints its four accumulator values. The entire kernel is one `asm volatile` block surrounded by variable declarations and a `printf`.

The instruction I targeted:

mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X .m16n8k32.row.col.f32.e2m1.e2m1.f32.ue8m0


Every piece of this string matters. `kind::mxf8f6f4` selects the instruction family that supports FP8, FP6, and FP4 types. `block_scale` enables hardware block scaling. `scale_vec::1X` means one scale factor per 32 elements along K. `m16n8k32` defines the tile shape: 16 rows, 8 columns, 32 elements along K. `e2m1.e2m1` specifies FP4 E2M1 for both A and B. `ue8m0` is the scale factor format.

### First surprise: the tile shape

I initially assumed the FP4 instruction would use shape m16n8k64. My reasoning: FP4 values are 4 bits, the hardware processes 32 bytes per step, 32 bytes / 0.5 bytes per element = 64 elements. So K=64.

Every attempt to compile with m16n8k64 failed:

error: "Incorrect instruction type specified for mma with shape m16n8k64"


I spent time going through CUTLASS source and NVIDIA forum posts before finding the answer. With `kind::mxf8f6f4`, the shape is always m16n8k32 regardless of element type. The K dimension in the shape name refers to the number of 8-bit containers, not the number of logical elements. For FP8, 32 containers hold 32 elements. For FP4, 32 containers still hold 32 elements — because each FP4 value is padded into an 8-bit container.

This was my first hint that the 8-bit container rule would be important. I did not fully appreciate it yet.

### Second surprise: the hardest bug

This one took the longest to find, and it is the kind of bug that inline PTX makes possible: everything runs, nothing crashes, the result is just wrong.

I encoded FP4 1.0 as the nibble `0010` (S=0, E=01, M=0), which is `0x2`. I packed eight of them into a 32-bit register: `0x22222222`. Both A and B filled with this value. Scale factors set to `0x7F7F7F7F` (UE8M0 for 1.0 — more on this format later). Accumulator initialized to zero.

The MMA ran. No error. The result: **2.0** on every lane.

The expected result was 1.0 × 1.0 × 32 elements = **32.0**. Off by a factor of 16.

I tested with other values. The results were internally consistent — doubling the input doubled the output — but always wrong by the same factor. This ruled out a random corruption. Something systematic was off in how I was encoding the data.

After several hours, I found the answer in a Discord thread quoting the PTX documentation:

> "When .kind is either of .kind::mxf8f6f4 or .kind::f8f6f4, the individual 4-bit floating point type elements must be packed in an 8-bit container. The matrix element of type .e2m1 resides in central 4 bits of the 8-bit container with padding in the upper 2 bits and lower 2 bits of the container."

Each FP4 value occupies one full byte. The 4 data bits sit at positions 5 through 2, with two zero bits above and two zero bits below:

bit: 7 6 5 4 3 2 1 0 0 0 S E E M 0 0


So FP4 1.0, which is nibble `0010`, becomes `00 0010 00` in binary — `0x08`, not `0x02`. When I packed `0x22` into each byte, the hardware was reading bits 5–2 of each byte, which gave it `1000` — the nibble for -0.0. The MMA was faithfully multiplying garbage.

With the correct encoding `0x08080808`, the MMA produced **32.0** across all lanes. Exactly right: 32 elements, each 1.0 × 1.0.

This is the kind of bug that no amount of `printf` debugging reveals at the PTX level. The instruction accepts anything. It does not validate your encoding. It just reads bits 5–2 and computes. If those bits are wrong, the result is wrong, silently. The only way I found it was to hand-compute what the hardware should see at each bit position, compare against what I was actually providing, and trace the discrepancy.

### The inline PTX syntax

For reference, here is the `asm volatile` block that runs the MMA. The syntax follows GCC inline assembly conventions — an instruction string, output operands, and input operands with register constraints:

```cuda
asm volatile(
    "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X"
    ".m16n8k32.row.col.f32.e2m1.e2m1.f32.ue8m0 "
    "{%0,%1,%2,%3}, "
    "{%4,%5,%6,%7}, "
    "{%8,%9}, "
    "{%10,%11,%12,%13}, "
    "{%14},{%15,%16}, "
    "{%17},{%18,%19};"
    : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
    : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
      "r"(B[0]), "r"(B[1]),
      "f"(D[0]), "f"(D[1]), "f"(D[2]), "f"(D[3]),
      "r"(sf_a), "h"((short)0), "h"((short)0),
      "r"(sf_b), "h"((short)0), "h"((short)0)
);
The "r" constraints are 32-bit integer registers (for the FP4 data packed into uint32), "f" are 32-bit float registers (for the FP32 accumulator), and "h" are 16-bit half registers (for the scale factor metadata, set to zero). The volatile keyword tells the compiler not to reorder, duplicate, or eliminate this block — critical for a warp-synchronous instruction that must execute exactly once, exactly where placed.

One important detail: the MMA is warp-synchronous. All 32 threads must participate. Launching the test kernel with a single thread produces all zeros — the instruction silently does nothing because the warp is incomplete. I had to use 32 threads to get results. This was another 20 minutes of confusion before I remembered the constraint.

What I learned from the test
Three things came out of this test, in order of how much time they cost me:

The 8-bit container rule. FP4 values are not packed two per byte on SM120 with kind::mxf8f6f4. Each value occupies a full byte, centered at bits 5–2. This encoding must be applied everywhere in the kernel — when quantizing activations, when repacking the score matrix between the two MMAs, when loading pre-quantized weights.

The shape rule. m16n8k32 counts containers, not elements. For FP4, 32 containers = 32 elements (not 64), because of the one-value-per-byte packing.

The throughput implication. Each 4-bit value wastes 4 bits of padding. SM120's FP4 throughput is effectively halved compared to what it could be if kind::mxf4nvf4 (true 4-bit packing) were available. This is a hardware limitation of consumer Blackwell that datacenter SM100 does not have.

5. Encoding FP32 to FP4 E2M1
Why we need an encoding function
The MMA test proved the hardware works when given correctly encoded data. But in that test, I hardcoded the values (0x08080808). In the real kernel, Q, K, and V arrive as FP16 or FP32 tensors from PyTorch. Every value must be converted to FP4 E2M1 at runtime, inside the kernel, in registers. No separate quantization pass, no round-trip through memory. I needed a __device__ function that takes a float and returns the corresponding 8-bit container.

The FP4 E2M1 format
4 bits: 1 sign bit (S), 2 exponent bits (E), 1 mantissa bit (M). The exponent bias is 1, from the standard IEEE formula: bias = 2^(exponent_bits - 1) - 1 = 2^1 - 1 = 1.

For normalized numbers (E > 0): value = (-1)^S × 1.M × 2^(E - bias). The notation 1.M means an implicit leading 1 followed by the mantissa bit. With M=0, 1.0 in binary = 1.0 in decimal. With M=1, 1.1 in binary = 1.5 (because the first digit after the binary point is 2^(-1) = 0.5).

For denormalized numbers (E = 0): value = (-1)^S × 0.M × 2^(1 - bias). The implicit leading bit becomes 0, allowing representation of values smaller than the smallest normalized number.

The complete table:

Nibble	S	E	M	Value
0000	0	00	0	+0.0
0001	0	00	1	+0.5
0010	0	01	0	+1.0
0011	0	01	1	+1.5
0100	0	10	0	+2.0
0101	0	10	1	+3.0
0110	0	11	0	+4.0
0111	0	11	1	+6.0
1000	1	00	0	-0.0
1001	1	00	1	-0.5
1010	1	01	0	-1.0
1011	1	01	1	-1.5
1100	1	10	0	-2.0
1101	1	10	1	-3.0
1110	1	11	0	-4.0
1111	1	11	1	-6.0
16 possible values. The distribution is non-uniform: resolution is finest near zero (steps of 0.5 between 0, 0.5, 1.0, 1.5) and coarsest at the extremes (a jump of 2.0 between 4.0 and 6.0). This is inherent to floating-point representation. With only 4 bits, it is the fundamental cost of aggressive quantization.

Building the function step by step
The value table is symmetric — the positive and negative halves are mirrors differing only in the sign bit. So I split the problem: extract the sign, find the correct magnitude on the absolute value, reattach the sign at the end.

Step 1: extract sign and magnitude.

__device__ uint8_t encode_fp4_e2m1(float val) {
    uint8_t sign = (val < 0.f) ? 1 : 0;
    float abs_val = fabsf(val);
    // ...
}
fabsf is a CUDA intrinsic for absolute value. After these two lines, sign is 0 or 1, and abs_val is the positive magnitude.

Step 2: find the nearest FP4 value by rounding. Given abs_val, I need to find which of the 8 positive FP4 values (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0) is closest. The standard approach is to compare against midpoints — the boundary where rounding switches from the lower value to the higher one:

Between	Midpoint
0.0 and 0.5	0.25
0.5 and 1.0	0.75
1.0 and 1.5	1.25
1.5 and 2.0	1.75
2.0 and 3.0	2.50
3.0 and 4.0	3.50
4.0 and 6.0	5.00
Values above 5.0 are clamped to 6.0, the maximum representable magnitude. This is implemented as a chain of comparisons starting from the top:

uint8_t nibble;
if      (abs_val >= 5.0f)  nibble = 0x7;  // 6.0
else if (abs_val >= 3.5f)  nibble = 0x6;  // 4.0
else if (abs_val >= 2.5f)  nibble = 0x5;  // 3.0
else if (abs_val >= 1.75f) nibble = 0x4;  // 2.0
else if (abs_val >= 1.25f) nibble = 0x3;  // 1.5
else if (abs_val >= 0.75f) nibble = 0x2;  // 1.0
else if (abs_val >= 0.25f) nibble = 0x1;  // 0.5
else                       nibble = 0x0;  // 0.0
On a GPU, this chain of comparisons against compile-time constants is efficient — the compiler converts it to predicated instructions without branch divergence. A lookup table alternative would require a shared memory or constant memory access, which is slower for such a short sequence.

The nibble values 0x0 through 0x7 are the 3-bit magnitude encodings from the table above: exponent and mantissa without the sign bit.

Step 3: attach the sign bit. The sign occupies bit 3 of the 4-bit nibble (the most significant bit of the FP4 value):

bit position:  3  2  1  0
               S  E  E  M
Shifting sign left by 3 and OR-ing it with the magnitude places the sign bit without disturbing the magnitude bits, because they occupy non-overlapping positions:

nibble |= (sign << 3);
Example: encoding -1.0. The magnitude nibble is 0010 (+1.0). The sign is 1. After 1 << 3 = 1000, the OR gives 0010 | 1000 = 1010 — the correct FP4 encoding for -1.0.

Step 4: place in the 8-bit container. As discovered during the MMA debugging, kind::mxf8f6f4 on SM120 requires each FP4 value at bits 5–2 of an 8-bit container. A left shift by 2:

return (uint8_t)(nibble << 2);
The final byte layout: 00 SEEM 00. Nibble 1010 (-1.0) shifted left by 2 becomes 00101000 = 0x28.

The complete function:

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
I tested the encoding function against known values, then fed the results into the MMA instruction to verify end-to-end correctness.

Input	Expected nibble	Expected byte	Observed
1.0	0010	0x08	0x08 ✓
-1.0	1010	0x28	0x28 ✓
6.0	0111	0x1C	0x1C ✓
1.2	0010 (rounded to 1.0)	0x08	0x08 ✓
The rounding test (1.2 → 1.0) confirms the midpoint logic: 1.2 < 1.25 (midpoint between 1.0 and 1.5), so the function rounds down correctly.

Packing into 32-bit registers
The MMA instruction expects operands as 32-bit registers. Each register holds 4 encoded FP4 values, one per byte. Packing uses shifts and OR to place each byte at the correct position:

uint8_t e0 = encode_fp4_e2m1(1.0f);
uint32_t packed = e0
    | (e0 << 8)
    | (e0 << 16)
    | (e0 << 24);
e0 goes to bits 7–0, e0 << 8 to bits 15–8, e0 << 16 to bits 23–16, e0 << 24 to bits 31–24. The four byte lanes do not overlap, so the OR assembles them cleanly. The result is 0x08080808, identical to what I hardcoded in the earlier MMA test.

End-to-end MMA validation
With the encoded and packed values replacing the hardcoded constants:

uint8_t e0 = encode_fp4_e2m1(1.0f);
uint32_t packed = e0
    | (e0 << 8)
    | (e0 << 16)
    | (e0 << 24);

uint32_t A[4] = {packed, packed, packed, packed};
uint32_t B[2] = {packed, packed};
The MMA produced 32.0 across all 32 lanes. Correct: 32 K-elements, each 1.0 × 1.0, summed.

I repeated the test with encode_fp4_e2m1(2.0f), which produced 128.0 (= 32 × 2.0 × 2.0). The full pipeline — encoding, packing, hardware MMA — works correctly for non-trivial values.

The encoding function now lives in common.h, shared between test files and the eventual kernel.

6. Block Scaling: Why the Encoding Function Is Not Enough
The saturation problem
The encoding function converts any float to FP4 E2M1. It works. But it has a hard ceiling that I had not fully internalized until I tested it with real-world-sized values.

encode(12.0) = 0x1C
encode(10.0) = 0x1C
Both produce the same byte. The function clamps everything above 5.0 to nibble 0111 (6.0), the maximum FP4 magnitude. 12.0 and 10.0, despite being 20% apart, become identical. If these were attention scores feeding into the second MMA, the model would weight both tokens identically when they should be weighted differently.

This is not a rounding problem — it is a saturation problem. The encoding function is correct, but the input exceeds the format's representable range.

The idea behind scale factors
The solution is to divide every value in a block by a common factor before encoding, so that the proportions between values are preserved even though the absolute magnitudes change.

I tested this concretely. With a block containing 12.0, 10.0, 3.0, and -7.0, dividing by 16 (the next power of two above the largest absolute value):

encode(12.0 / 16.0) = 0x08   (1.0)
encode(10.0 / 16.0) = 0x04   (0.5)
encode( 3.0 / 16.0) = 0x00   (0.0)
encode(-7.0 / 16.0) = 0x24   (-0.5)
Without the scale factor, 12.0 and 10.0 both mapped to 0x1C — indistinguishable. With it, they map to 0x08 and 0x04. The proportions are not perfectly preserved (the true ratio 10/12 ≈ 0.83 rounds to 0.5/1.0 = 0.5), but the values are at least distinct. In a 4-bit format with 16 possible outputs, this is the best you can do.

The hardware needs to know what divisor was used, so it can undo the division during the MMA. If we divided by 16, the MMA result must be multiplied by 16 to restore correct magnitude. This multiplication happens automatically inside the Tensor Core — the scale factor is a register operand alongside the data, and the hardware applies it during the multiply-accumulate. No extra instructions.

Why the scale factor must be a power of two
The MMA instruction accepts scale factors in UE8M0 format: 8-bit unsigned, 8 exponent bits, zero mantissa bits. No sign, no fractional part. The value it represents is 2^(stored_byte - 127), where 127 is the exponent bias — the same convention as FP32, borrowed from the OCP Microscaling Formats (MX) specification.

Because there is no mantissa, UE8M0 can only encode powers of two: 1, 2, 4, 8, 16, and so on up to 2^127 (and down to 2^-127). You cannot store 12.0 as a scale factor. You store either 8 (2^3) or 16 (2^4).

This constraint is deliberate. Multiplying by a power of two is an exponent shift in IEEE 754 representation — zero additional logic inside the Tensor Core. A non-power-of-two scale would require a real multiply, eating into throughput.

Rounding direction: up, not down
If the maximum absolute value in a block is 12.0, should the scale factor be 8 or 16?

Rounding down to 8: dividing by 8 gives 12.0/8 = 1.5, which fits in FP4. But a block containing 49.0 would give 49.0/8 = 6.125, exceeding FP4's maximum of 6.0 — saturation. Rounding down risks overflow.

Rounding up to 16: dividing by 16 gives 12.0/16 = 0.75. Nothing can exceed 6.0 because the scale is at least as large as the block maximum. Resolution is reduced (values pushed closer to zero), but overflow is impossible.

The MX standard rounds up. I follow the same convention.

Computing the UE8M0 byte
Three operations:

Find the maximum absolute value in the block. Compute the smallest integer exponent such that 2^exponent is greater than or equal to max_abs: exponent = ceil(log2(max_abs)). Add the bias: byte = exponent + 127.

Concrete example: max_abs = 12.0. log2f(12.0) = 3.58. ceilf(3.58) = 4. Exponent is 4. Stored byte: 4 + 127 = 131.

Verification: hardware reads 131, subtracts the bias (131 - 127 = 4), computes 2^4 = 16. The MMA multiplies its result by 16. Correct.

Another example: max_abs = 1.0 (the trivial case, everything already in FP4 range). log2f(1.0) = 0. ceilf(0) = 0. Stored byte: 0 + 127 = 127 = 0x7F. This is exactly the value I hardcoded in sf_a and sf_b throughout every earlier MMA test. 0x7F in UE8M0 means "scale factor = 1.0" — no scaling.

The device function
__device__ uint8_t compute_scale_ue8m0(float* block, int size) {
    float max_abs = 0.0f;
    for (int i = 0; i < size; i++) {
        float a = fabsf(block[i]);
        if (a > max_abs) max_abs = a;
    }

    int exponent = (int)ceilf(log2f(max_abs));
    uint8_t ue8m0 = (uint8_t)(exponent + 127);
    return ue8m0;
}
The loop finds the maximum absolute value — fabsf because -12.0 and +12.0 have the same impact on the range. log2f + ceilf round up to the next power of two. Adding 127 produces the biased exponent the hardware expects.

End-to-end MMA validation with scale factors
To verify the full pipeline, I ran a test where every element of A is 8.0, B is 1.0, scale_A = 8 (UE8M0 byte = 3 + 127 = 130 = 0x82), scale_B = 1 (UE8M0 byte = 127 = 0x7F).

Before encoding, A values are divided by the scale: 8.0 / 8.0 = 1.0. B values: 1.0 / 1.0 = 1.0. Both encode to 0x08 (FP4 1.0), packed into 0x08080808.

The MMA computes the dot product: 1.0 × 1.0 × 32 = 32. Hardware multiplies by scale_A × scale_B = 8 × 1 = 8. Final result: 32 × 8 = 256.0.

All 32 lanes reported 256.0. The scale factor is correctly applied by the hardware.

8.0 is a power of two, so the scale factor lands exactly on the block maximum — no rounding loss. In practice, non-power-of-two maxima introduce a gap between the actual maximum and the scale factor. A block with max 12.0 uses a scale of 16, meaning FP4's range of 0 to 6.0 maps to 0 to 96.0, while the data only occupies 0 to 12.0. Three quarters of the representable range is wasted. This is the cost of restricting scale factors to powers of two.

Block size: precision versus overhead
On SM120 with kind::mxf8f6f4 and scale_vec::1X, each scale factor covers 32 consecutive elements along K. This is not configurable — the instruction dictates it.

Consider a block of 32 values where one outlier is 100.0 and the rest are around 0.1. The scale factor will be 128 (2^7). After dividing by 128, the outlier becomes 0.78 (rounds to 1.0 in FP4), but the others become 0.1/128 ≈ 0.0008, which rounds to 0.0. The outlier survives; everything else is erased.

A smaller block size (16 elements, available on SM100 with scale_vec::2X) would let the non-outlier region have its own scale factor, preserving more information. On SM120, we cannot do this. It is one more reason to care about value distributions within each 32-element block — particularly after softmax, where most values are small and a few dominate.

What we have now
Two validated building blocks:

encode_fp4_e2m1(float val) — converts a float to an 8-bit FP4 E2M1 container, centered at bits 5–2 as required by kind::mxf8f6f4 on SM120.

compute_scale_ue8m0(float* block, int size) — computes the UE8M0 scale factor for a block of floats, rounding up to the next power of two.

Together they form the quantization pass: find the scale, divide, encode. The MMA receives both the encoded data and the scale factor, and produces the correctly scaled FP32 result.

Both functions live in common.h. The test file and the final kernel include the same header.

Next: loading Q, K, and V from global memory into shared memory using asynchronous copies (cp.async), where the kernel starts to look like a real attention implementation rather than a collection of unit tests.
