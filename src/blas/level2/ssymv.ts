/**
 * SSYMV - Symmetric matrix-vector multiply (single precision)
 * 
 * Performs the operation: y = alpha * A * x + beta * y
 * where A is a symmetric n x n matrix, x and y are n-element vectors,
 * and alpha and beta are scalars.
 * 
 * Only the upper or lower triangular part of A is referenced.
 * 
 * This is a WebGPU implementation of the reference BLAS SSYMV function.
 */

/// <reference types="@webgpu/types" />

import { getWebGPUContext } from '../../webgpu/context';
import ssymvShader from '../../shaders/ssymv.wgsl';

/**
 * Uplo parameter for SYMV functions
 */
export type SYMVUplo = 'U' | 'L';

/**
 * SSYMV parameters
 */
export interface SSYMVParams {
  /** Specifies whether the upper ('U') or lower ('L') triangular part of A is to be referenced */
  uplo: SYMVUplo;
  /** Order of the matrix A (n x n) */
  n: number;
  /** Scalar alpha */
  alpha: number;
  /** Symmetric matrix A stored in column-major order (n x n) */
  A: Float32Array;
  /** Leading dimension of A (usually n, must be >= n) */
  lda: number;
  /** Input vector x */
  x: Float32Array;
  /** Increment/stride between elements in x (default: 1) */
  incx?: number;
  /** Scalar beta */
  beta: number;
  /** Input/Output vector y (modified in-place) */
  y: Float32Array;
  /** Increment/stride between elements in y (default: 1) */
  incy?: number;
}

/**
 * Symmetric matrix-vector multiply using WebGPU (single precision)
 * Performs: y = alpha * A * x + beta * y
 */
export async function ssymv(params: SSYMVParams): Promise<void> {
  const { uplo, n, alpha, A, lda, x, incx = 1, beta, y, incy = 1 } = params;
  
  // Validate inputs
  if (n <= 0) return;
  if (incx === 0 || incy === 0) return;
  if (lda < n) throw new Error('SSYMV: lda must be >= n');
  if (A.length < lda * n) throw new Error('SSYMV: Matrix A is too small');
  
  // Validate vector sizes
  if (x.length < (n - 1) * Math.abs(incx) + 1) {
    throw new Error('SSYMV: Vector x is too small');
  }
  if (y.length < (n - 1) * Math.abs(incy) + 1) {
    throw new Error('SSYMV: Vector y is too small');
  }
  
  const context = await getWebGPUContext();
  const device = context.getDevice();
  
  // Create buffers
  const ABuffer = context.createBuffer(A);
  const xBuffer = context.createBuffer(x);
  const yData = new Float32Array(y);
  const yBuffer = context.createBuffer(yData);
  
  // Create uniform buffer for parameters (16-byte aligned)
  const paramsData = new ArrayBuffer(32); // 5 u32s/i32s + padding for 16-byte alignment
  const paramsView = new DataView(paramsData);
  paramsView.setUint32(0, n, true);
  paramsView.setUint32(4, lda, true);
  paramsView.setInt32(8, incx, true);
  paramsView.setInt32(12, incy, true);
  paramsView.setUint32(16, uplo === 'U' ? 1 : 0, true); // 1 for upper, 0 for lower
  // Padding bytes for alignment
  
  const paramsBuffer = device.createBuffer({
    size: paramsData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint8Array(paramsBuffer.getMappedRange()).set(new Uint8Array(paramsData));
  paramsBuffer.unmap();
  
  // Create uniform buffers for alpha and beta (16-byte aligned each)
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
  
  const betaData = new ArrayBuffer(16); // f32 + padding
  const betaView = new DataView(betaData);
  betaView.setFloat32(0, beta, true);
  
  const betaBuffer = device.createBuffer({
    size: betaData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint8Array(betaBuffer.getMappedRange()).set(new Uint8Array(betaData));
  betaBuffer.unmap();
  
  // Create compute pipeline
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: device.createShaderModule({
        code: ssymvShader,
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
        resource: { buffer: yBuffer },
      },
      {
        binding: 3,
        resource: { buffer: paramsBuffer },
      },
      {
        binding: 4,
        resource: { buffer: alphaBuffer },
      },
      {
        binding: 5,
        resource: { buffer: betaBuffer },
      },
    ],
  });
  
  // Execute computation
  const commandEncoder = device.createCommandEncoder();
  const computePass = commandEncoder.beginComputePass();
  computePass.setPipeline(pipeline);
  computePass.setBindGroup(0, bindGroup);
  
  // Dispatch workgroups for n elements
  const WORKGROUP_SIZE = 256;
  const numWorkgroups = Math.ceil(n / WORKGROUP_SIZE);
  computePass.dispatchWorkgroups(numWorkgroups);
  computePass.end();
  
  device.queue.submit([commandEncoder.finish()]);
  
  // Read results
  const yResultData = await context.readBuffer(yBuffer, yData.byteLength);
  const yResult = new Float32Array(yResultData);
  
  // Cleanup
  ABuffer.destroy();
  xBuffer.destroy();
  yBuffer.destroy();
  paramsBuffer.destroy();
  alphaBuffer.destroy();
  betaBuffer.destroy();
  
  // Copy results back to original array
  y.set(yResult.slice(0, y.length));
}
