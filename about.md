---
layout: page
title: About
---

I'm Florian Mattana, a software engineer specializing in C++ and CUDA.

This blog is a deep dive into GPU kernel engineering: implementing, 
profiling, and optimizing foundational deep learning kernels in CUDA 
from scratch. 
Every kernel is benchmarked against production libraries 
like cuBLAS using Nsight Compute, with full analysis of memory 
throughput, compute utilization, and bottleneck identification.

Topics covered include GEMM optimization (tiling, register blocking, 
vectorized loads, double buffering), parallel reduction, prefix scan, 
Tensor Cores (WMMA and PTX), softmax, Flash Attention, and multi-GPU 
communication.



