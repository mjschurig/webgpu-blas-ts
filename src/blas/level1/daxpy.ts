/**
 * DAXPY - Compute y = alpha * x + y
 * 
 * Performs the operation y := alpha * x + y where alpha is a scalar,
 * x and y are vectors. This is a WebGPU implementation of the reference BLAS DAXPY function.
 */

/// <reference types="@webgpu/types" />

import { getWebGPUContext } from '../../webgpu/context';
import daxpyShader from '../../shaders/daxpy.wgsl';

/**
 * DAXPY parameters
 */
export interface DAXPYParams {
  /** Number of elements in the vectors */
  n: number;
  /** Scalar multiplier alpha */
  alpha: number;
  /** Input vector x */
  dx: Float64Array;
  /** Increment/stride between elements in x (default: 1) */
  incx?: number;
  /** Input/output vector y */
  dy: Float64Array;
  /** Increment/stride between elements in y (default: 1) */
  incy?: number;
}

/**
 * Compute y = alpha * x + y using WebGPU
 * Modifies the y vector in-place.
 */
export async function daxpy(params: DAXPYParams): Promise<void> {
  const { n, alpha, dx, dy, incx = 1, incy = 1 } = params;
  
  // Validate inputs
  if (n <= 0) return;
  if (incx === 0 || incy === 0) return;
  if (dx.length === 0 || dy.length === 0) return;
  
  // Early return if alpha is zero
  if (alpha === 0.0) return;
  
  const context = await getWebGPUContext();
  const device = context.getDevice();
  
  // Create buffers using context methods
  const xBuffer = context.createBuffer(dx);
  
  // Create a copy of y data for GPU processing
  const yData = new Float64Array(dy);
  const yBuffer = context.createBuffer(yData);
  
  // Create uniform buffer for parameters
  const paramsData = new ArrayBuffer(12); // 3 u32s/i32s
  const paramsView = new DataView(paramsData);
  paramsView.setUint32(0, n, true);
  paramsView.setInt32(4, incx, true);
  paramsView.setInt32(8, incy, true);
  
  const paramsBuffer = device.createBuffer({
    size: paramsData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint8Array(paramsBuffer.getMappedRange()).set(new Uint8Array(paramsData));
  paramsBuffer.unmap();
  
  // Create uniform buffer for alpha
  const alphaData = new ArrayBuffer(8); // f64
  const alphaView = new DataView(alphaData);
  alphaView.setFloat64(0, alpha, true);
  
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
        code: daxpyShader,
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
        resource: { buffer: xBuffer },
      },
      {
        binding: 1,
        resource: { buffer: yBuffer },
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
  
  // Dispatch workgroups - calculate number needed for n elements
  const WORKGROUP_SIZE = 256;
  const numWorkgroups = Math.ceil(n / WORKGROUP_SIZE);
  computePass.dispatchWorkgroups(numWorkgroups);
  computePass.end();
  
  device.queue.submit([commandEncoder.finish()]);
  
  // Read result
  const resultData = await context.readBuffer(yBuffer, yData.byteLength);
  const result = new Float64Array(resultData);
  
  // Cleanup
  xBuffer.destroy();
  yBuffer.destroy();
  paramsBuffer.destroy();
  alphaBuffer.destroy();
  
  // Copy result back to original dy array
  dy.set(result.slice(0, dy.length));
}
