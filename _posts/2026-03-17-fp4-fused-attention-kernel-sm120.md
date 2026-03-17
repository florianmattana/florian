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
