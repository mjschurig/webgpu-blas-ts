/**
 * SGER - General rank-1 update (single precision)
 * 
 * Performs the operation: A := alpha * x * y^T + A
 * where A is an m x n matrix, x is an m-element vector, y is an n-element vector,
 * and alpha is a scalar.
 * 
 * This is a WebGPU implementation of the reference BLAS SGER function.
 */

/// <reference types="@webgpu/types" />

import { getWebGPUContext } from '../../webgpu/context';
import sgerShader from '../../shaders/sger.wgsl';

/**
 * SGER parameters
 */
export interface SGERParams {
  /** Number of rows of matrix A */
  m: number;
  /** Number of columns of matrix A */
  n: number;
  /** Scalar alpha */
  alpha: number;
  /** Vector x (m elements) */
  x: Float32Array;
  /** Increment/stride between elements in x (default: 1) */
  incx?: number;
  /** Vector y (n elements) */
  y: Float32Array;
  /** Increment/stride between elements in y (default: 1) */
  incy?: number;
  /** Matrix A stored in column-major order (m x n) - modified in-place */
  A: Float32Array;
  /** Leading dimension of A (usually m, must be >= m) */
  lda: number;
}

/**
 * General rank-1 update using WebGPU (single precision)
 * Performs: A := alpha * x * y^T + A
 */
export async function sger(params: SGERParams): Promise<void> {
  const { m, n, alpha, x, incx = 1, y, incy = 1, A, lda } = params;
  
  // Validate inputs
  if (m <= 0 || n <= 0) return;
  if (incx === 0 || incy === 0) return;
  if (lda < m) throw new Error('SGER: lda must be >= m');
  if (A.length < lda * n) throw new Error('SGER: Matrix A is too small');
  
  // Validate vector sizes
  if (x.length < (m - 1) * Math.abs(incx) + 1) {
    throw new Error('SGER: Vector x is too small');
  }
  if (y.length < (n - 1) * Math.abs(incy) + 1) {
    throw new Error('SGER: Vector y is too small');
  }
  
  // Early return if alpha is zero (no operation needed)
  if (alpha === 0) return;
  
  const context = await getWebGPUContext();
  const device = context.getDevice();
  
  // Create buffers
  const AData = new Float32Array(A);
  const ABuffer = context.createBuffer(AData);
  const xBuffer = context.createBuffer(x);
  const yBuffer = context.createBuffer(y);
  
  // Create uniform buffer for parameters (16-byte aligned)
  const paramsData = new ArrayBuffer(32); // 5 u32s/i32s + padding for 16-byte alignment
  const paramsView = new DataView(paramsData);
  paramsView.setUint32(0, m, true);
  paramsView.setUint32(4, n, true);
  paramsView.setUint32(8, lda, true);
  paramsView.setInt32(12, incx, true);
  paramsView.setInt32(16, incy, true);
  // Padding bytes for alignment
  
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
        code: sgerShader,
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
  const numWorkgroupsY = Math.ceil(m / WORKGROUP_SIZE_Y);
  computePass.dispatchWorkgroups(numWorkgroupsX, numWorkgroupsY);
  computePass.end();
  
  device.queue.submit([commandEncoder.finish()]);
  
  // Read results
  const AResultData = await context.readBuffer(ABuffer, AData.byteLength);
  const AResult = new Float32Array(AResultData);
  
  // Cleanup
  ABuffer.destroy();
  xBuffer.destroy();
  yBuffer.destroy();
  paramsBuffer.destroy();
  alphaBuffer.destroy();
  
  // Copy results back to original array
  A.set(AResult.slice(0, A.length));
}
