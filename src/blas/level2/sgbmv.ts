/**
 * SGBMV - General banded matrix-vector multiply (single precision)
 * 
 * Performs the operation: y = alpha * op(A) * x + beta * y
 * where op(A) is either A or A^T (transpose) based on the trans parameter,
 * A is an m x n banded matrix with kl sub-diagonals and ku super-diagonals,
 * x and y are vectors, and alpha and beta are scalars.
 * 
 * This is a WebGPU implementation of the reference BLAS SGBMV function.
 */

/// <reference types="@webgpu/types" />

import { getWebGPUContext } from '../../webgpu/context';
import sgbmvShader from '../../shaders/sgbmv.wgsl';

/**
 * SGBMV parameters
 */
export interface SGBMVParams {
  /** Transpose operation: 'N' = no transpose, 'T' = transpose */
  trans: 'N' | 'T';
  /** Number of rows of matrix A */
  m: number;
  /** Number of columns of matrix A */
  n: number;
  /** Number of sub-diagonals of A */
  kl: number;
  /** Number of super-diagonals of A */
  ku: number;
  /** Scalar alpha */
  alpha: number;
  /** Banded matrix A stored in banded storage format */
  A: Float32Array;
  /** Leading dimension of A (must be >= kl + ku + 1) */
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
 * General banded matrix-vector multiply using WebGPU (single precision)
 * Performs: y = alpha * op(A) * x + beta * y
 */
export async function sgbmv(params: SGBMVParams): Promise<void> {
  const { trans, m, n, kl, ku, alpha, A, lda, x, incx = 1, beta, y, incy = 1 } = params;
  
  // Validate inputs
  if (m <= 0 || n <= 0) return;
  if (kl < 0 || ku < 0) throw new Error('SGBMV: kl and ku must be >= 0');
  if (incx === 0 || incy === 0) return;
  if (lda < kl + ku + 1) throw new Error('SGBMV: lda must be >= kl + ku + 1');
  if (A.length < lda * n) throw new Error('SGBMV: Banded matrix A is too small');
  
  // Validate vector sizes based on transpose operation
  const x_size = trans === 'N' ? n : m;  // x size depends on operation
  const y_size = trans === 'N' ? m : n;  // y size depends on operation
  
  if (x.length < (x_size - 1) * Math.abs(incx) + 1) {
    throw new Error('SGBMV: Vector x is too small');
  }
  if (y.length < (y_size - 1) * Math.abs(incy) + 1) {
    throw new Error('SGBMV: Vector y is too small');
  }
  
  const context = await getWebGPUContext();
  const device = context.getDevice();
  
  // Create buffers
  const ABuffer = context.createBuffer(A);
  const xBuffer = context.createBuffer(x);
  const yData = new Float32Array(y);
  const yBuffer = context.createBuffer(yData);
  
  // Create uniform buffer for parameters (16-byte aligned)
  const paramsData = new ArrayBuffer(48); // 8 u32s + padding for 16-byte alignment
  const paramsView = new DataView(paramsData);
  paramsView.setUint32(0, m, true);
  paramsView.setUint32(4, n, true);
  paramsView.setUint32(8, kl, true);
  paramsView.setUint32(12, ku, true);
  paramsView.setUint32(16, lda, true);
  paramsView.setInt32(20, incx, true);
  paramsView.setInt32(24, incy, true);
  paramsView.setUint32(28, trans === 'T' ? 1 : 0, true); // 1 for transpose, 0 for no transpose
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
        code: sgbmvShader,
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
  
  // Dispatch workgroups based on output vector size
  const WORKGROUP_SIZE = 256;
  const outputSize = trans === 'N' ? m : n;
  const numWorkgroups = Math.ceil(outputSize / WORKGROUP_SIZE);
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
