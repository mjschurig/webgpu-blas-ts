/**
 * STRMV - Triangular matrix-vector multiply (single precision)
 * 
 * Performs the operation: x := A * x or x := A^T * x
 * where A is a triangular n x n matrix and x is an n-element vector.
 * 
 * Only the upper or lower triangular part of A is referenced,
 * and the matrix can be unit or non-unit triangular.
 * 
 * This is a WebGPU implementation of the reference BLAS STRMV function.
 */

/// <reference types="@webgpu/types" />

import { getWebGPUContext } from '../../webgpu/context';
import strmvShader from '../../shaders/strmv.wgsl';

/**
 * Uplo parameter for triangular matrix functions
 */
export type TRMVUplo = 'U' | 'L';

/**
 * Trans parameter for triangular matrix functions
 */
export type TRMVTrans = 'N' | 'T';

/**
 * Diag parameter for triangular matrix functions
 */
export type TRMVDiag = 'N' | 'U';

/**
 * STRMV parameters
 */
export interface STRMVParams {
  /** Specifies whether A is upper ('U') or lower ('L') triangular */
  uplo: TRMVUplo;
  /** Specifies the operation: 'N' = A * x, 'T' = A^T * x */
  trans: TRMVTrans;
  /** Specifies whether A is unit ('U') or non-unit ('N') triangular */
  diag: TRMVDiag;
  /** Order of the matrix A (n x n) */
  n: number;
  /** Triangular matrix A stored in column-major order (n x n) */
  A: Float32Array;
  /** Leading dimension of A (usually n, must be >= n) */
  lda: number;
  /** Vector x (n elements) - modified in-place */
  x: Float32Array;
  /** Increment/stride between elements in x (default: 1) */
  incx?: number;
}

/**
 * Triangular matrix-vector multiply using WebGPU (single precision)
 * Performs: x := A * x or x := A^T * x
 */
export async function strmv(params: STRMVParams): Promise<void> {
  const { uplo, trans, diag, n, A, lda, x, incx = 1 } = params;
  
  // Validate inputs
  if (n <= 0) return;
  if (incx === 0) return;
  if (lda < n) throw new Error('STRMV: lda must be >= n');
  if (A.length < lda * n) throw new Error('STRMV: Matrix A is too small');
  
  // Validate vector size
  if (x.length < (n - 1) * Math.abs(incx) + 1) {
    throw new Error('STRMV: Vector x is too small');
  }
  
  const context = await getWebGPUContext();
  const device = context.getDevice();
  
  // Create buffers
  const ABuffer = context.createBuffer(A);
  const xInBuffer = context.createBuffer(x);
  const xOutData = new Float32Array(x.length);
  const xOutBuffer = context.createBuffer(xOutData);
  
  // Create uniform buffer for parameters (16-byte aligned)
  const paramsData = new ArrayBuffer(32); // 6 u32s + padding for 16-byte alignment
  const paramsView = new DataView(paramsData);
  paramsView.setUint32(0, n, true);
  paramsView.setUint32(4, lda, true);
  paramsView.setInt32(8, incx, true);
  paramsView.setUint32(12, uplo === 'U' ? 1 : 0, true); // 1 for upper, 0 for lower
  paramsView.setUint32(16, trans === 'T' ? 1 : 0, true); // 1 for transpose, 0 for no transpose
  paramsView.setUint32(20, diag === 'U' ? 1 : 0, true); // 1 for unit, 0 for non-unit
  // Padding bytes for alignment
  
  const paramsBuffer = device.createBuffer({
    size: paramsData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint8Array(paramsBuffer.getMappedRange()).set(new Uint8Array(paramsData));
  paramsBuffer.unmap();
  
  // Create compute pipeline
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: device.createShaderModule({
        code: strmvShader,
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
        resource: { buffer: xInBuffer },
      },
      {
        binding: 2,
        resource: { buffer: xOutBuffer },
      },
      {
        binding: 3,
        resource: { buffer: paramsBuffer },
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
  const xResultData = await context.readBuffer(xOutBuffer, xOutData.byteLength);
  const xResult = new Float32Array(xResultData);
  
  // Cleanup
  ABuffer.destroy();
  xInBuffer.destroy();
  xOutBuffer.destroy();
  paramsBuffer.destroy();
  
  // Copy results back to original array
  x.set(xResult.slice(0, x.length));
}
