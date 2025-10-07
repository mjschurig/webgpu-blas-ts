/**
 * SSYR - Symmetric rank-1 update (single precision)
 * 
 * Performs the operation: A := alpha * x * x^T + A
 * where A is a symmetric n x n matrix, x is an n-element vector,
 * and alpha is a scalar.
 * 
 * Only the upper or lower triangular part of A is modified.
 * 
 * This is a WebGPU implementation of the reference BLAS SSYR function.
 */

/// <reference types="@webgpu/types" />

import { getWebGPUContext } from '../../webgpu/context';
import ssyrShader from '../../shaders/ssyr.wgsl';

/**
 * SSYR parameters
 */
export interface SSYRParams {
  /** Specifies whether the upper ('U') or lower ('L') triangular part of A is to be modified */
  uplo: 'U' | 'L';
  /** Order of the matrix A (n x n) */
  n: number;
  /** Scalar alpha */
  alpha: number;
  /** Vector x (n elements) */
  x: Float32Array;
  /** Increment/stride between elements in x (default: 1) */
  incx?: number;
  /** Symmetric matrix A stored in column-major order (n x n) - modified in-place */
  A: Float32Array;
  /** Leading dimension of A (usually n, must be >= n) */
  lda: number;
}

/**
 * Symmetric rank-1 update using WebGPU (single precision)
 * Performs: A := alpha * x * x^T + A
 */
export async function ssyr(params: SSYRParams): Promise<void> {
  const { uplo, n, alpha, x, incx = 1, A, lda } = params;
  
  // Validate inputs
  if (n <= 0) return;
  if (incx === 0) return;
  if (lda < n) throw new Error('SSYR: lda must be >= n');
  if (A.length < lda * n) throw new Error('SSYR: Matrix A is too small');
  
  // Validate vector size
  if (x.length < (n - 1) * Math.abs(incx) + 1) {
    throw new Error('SSYR: Vector x is too small');
  }
  
  // Early return if alpha is zero (no operation needed)
  if (alpha === 0) return;
  
  const context = await getWebGPUContext();
  const device = context.getDevice();
  
  // Create buffers
  const AData = new Float32Array(A);
  const ABuffer = context.createBuffer(AData);
  const xBuffer = context.createBuffer(x);
  
  // Create uniform buffer for parameters (16-byte aligned)
  const paramsData = new ArrayBuffer(16); // 4 u32s/i32s for 16-byte alignment
  const paramsView = new DataView(paramsData);
  paramsView.setUint32(0, n, true);
  paramsView.setUint32(4, lda, true);
  paramsView.setInt32(8, incx, true);
  paramsView.setUint32(12, uplo === 'U' ? 1 : 0, true); // 1 for upper, 0 for lower
  
  const paramsBuffer = device.createBuffer({
    size: paramsData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint8Array(paramsBuffer.getMappedRange()).set(new Uint8Array(paramsData));
  paramsBuffer.unmap();
  
  // Create uniform buffer for alpha (16-byte aligned)
  const alphaData = new ArrayBuffer(16); // f32 + padding
  const alphaView = new DataView(alphaData);
  alphaView.setFloat32(0, alpha, true);
  
  const alphaBuffer = device.createBuffer({
    size: alphaData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint8Array(alphaBuffer.getMappedRange()).set(new Uint8Array(alphaData));
  alphaBuffer.unmap();
  
  // Create compute pipeline
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: device.createShaderModule({
        code: ssyrShader,
      }),
      entryPoint: 'main',
    },
  });
  
  // Create bind group
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      {
        binding: 0,
        resource: { buffer: ABuffer },
      },
      {
        binding: 1,
        resource: { buffer: xBuffer },
      },
      {
        binding: 2,
        resource: { buffer: paramsBuffer },
      },
      {
        binding: 3,
        resource: { buffer: alphaBuffer },
      },
    ],
  });
  
  // Execute computation
  const commandEncoder = device.createCommandEncoder();
  const computePass = commandEncoder.beginComputePass();
  computePass.setPipeline(pipeline);
  computePass.setBindGroup(0, bindGroup);
  
  // Dispatch 2D workgroups (8x8 threads per workgroup)
  const WORKGROUP_SIZE_X = 8;
  const WORKGROUP_SIZE_Y = 8;
  const numWorkgroupsX = Math.ceil(n / WORKGROUP_SIZE_X);
  const numWorkgroupsY = Math.ceil(n / WORKGROUP_SIZE_Y);
  computePass.dispatchWorkgroups(numWorkgroupsX, numWorkgroupsY);
  computePass.end();
  
  device.queue.submit([commandEncoder.finish()]);
  
  // Read results
  const AResultData = await context.readBuffer(ABuffer, AData.byteLength);
  const AResult = new Float32Array(AResultData);
  
  // Cleanup
  ABuffer.destroy();
  xBuffer.destroy();
  paramsBuffer.destroy();
  alphaBuffer.destroy();
  
  // Copy results back to original array
  A.set(AResult.slice(0, A.length));
}
