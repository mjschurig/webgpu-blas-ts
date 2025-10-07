# WebGPU BLAS Shaders

This directory contains WGSL (WebGPU Shading Language) compute shaders for BLAS operations.

## Level 1 BLAS Operations

### ASUM - Sum of Absolute Values

**Single Precision (SASUM)** 
- `sasum_stage1.wgsl` - First stage parallel reduction within workgroups (f32)
- `sasum_stage2.wgsl` - Second stage final reduction across workgroups (f32)

### AXPY - Compute y = alpha * x + y

**Single Precision (SAXPY)**
- `saxpy.wgsl` - Parallel vector operation using f32 precision

## Implementation Notes

### Workgroup Size
All shaders use a standard workgroup size of 256 threads for optimal GPU utilization across different hardware.

### Two-Stage Reduction Pattern
The ASUM operations use a two-stage reduction pattern inspired by ROCm BLAS:
1. **Stage 1**: Each workgroup processes multiple elements per thread (WIN=4) and reduces within workgroup shared memory
2. **Stage 2**: Final reduction across workgroup results (only needed when multiple workgroups are used)

### Precision Handling
- Single precision shaders use `f32` type and process `Float32Array` data  
- Buffer sizes are automatically determined based on precision (4 bytes for f32)
- Note: WebGPU has limited support for double precision, so this library focuses on single precision only

### Stride Support
All operations support strided access patterns via `incx` and `incy` parameters, enabling efficient processing of non-contiguous data.