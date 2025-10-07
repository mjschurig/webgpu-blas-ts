/**
 * SAXPY - Compute y = alpha * x + y (single precision)
 * 
 * Performs the operation y := alpha * x + y where alpha is a scalar,
 * x and y are vectors. This is a WebGPU implementation of the reference BLAS SAXPY function.
 */

/// <reference types="@webgpu/types" />

import { getWebGPUContext } from '../../webgpu/context';
import saxpyShader from '../../shaders/saxpy.wgsl';

/**
 * SAXPY parameters
 */
export interface SAXPYParams {
  /** Number of elements in the vectors */
  n: number;
  /** Scalar multiplier alpha */
  alpha: number;
  /** Input vector x */
  sx: Float32Array;
  /** Increment/stride between elements in x (default: 1) */
  incx?: number;
  /** Input/output vector y */
  sy: Float32Array;
  /** Increment/stride between elements in y (default: 1) */
  incy?: number;
}

/**
 * Compute y = alpha * x + y using WebGPU (single precision)
 * Modifies the y vector in-place.
 */
export async function saxpy(params: SAXPYParams): Promise<void> {
  const { n, alpha, sx, sy, incx = 1, incy = 1 } = params;
  
  // Validate inputs
  if (n <= 0) return;
  if (incx === 0 || incy === 0) return;
  if (sx.length === 0 || sy.length === 0) return;
  
  // Early return if alpha is zero
  if (alpha === 0.0) return;
  
  const context = await getWebGPUContext();
  const device = context.getDevice();
  
  // Create buffers using context methods
  const xBuffer = context.createBuffer(sx);
  
  // Create a copy of y data for GPU processing
  const yData = new Float32Array(sy);
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
  
  // Create uniform buffer for alpha (f32)
  const alphaData = new ArrayBuffer(4); // f32
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
        code: saxpyShader,
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
  const result = new Float32Array(resultData);
  
  // Cleanup
  xBuffer.destroy();
  yBuffer.destroy();
  paramsBuffer.destroy();
  alphaBuffer.destroy();
  
  // Copy result back to original sy array
  sy.set(result.slice(0, sy.length));
}
