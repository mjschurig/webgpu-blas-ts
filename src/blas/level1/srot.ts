/**
 * SROT - Apply plane rotation to vectors (single precision)
 * 
 * Applies a plane (Givens) rotation to vectors x and y:
 * temp_x = c * x + s * y
 * temp_y = c * y - s * x
 * x = temp_x, y = temp_y
 * 
 * This is a WebGPU implementation of the reference BLAS SROT function.
 */

/// <reference types="@webgpu/types" />

import { getWebGPUContext } from '../../webgpu/context';
import srotShader from '../../shaders/srot.wgsl';

/**
 * SROT parameters
 */
export interface SROTParams {
  /** Number of elements in the vectors */
  n: number;
  /** First vector x (modified in-place) */
  sx: Float32Array;
  /** Increment/stride between elements in x (default: 1) */
  incx?: number;
  /** Second vector y (modified in-place) */
  sy: Float32Array;
  /** Increment/stride between elements in y (default: 1) */
  incy?: number;
  /** Cosine of rotation angle */
  c: number;
  /** Sine of rotation angle */
  s: number;
}

/**
 * Apply plane rotation to two vectors using WebGPU (single precision)
 * Modifies both vectors in-place.
 */
export async function srot(params: SROTParams): Promise<void> {
  const { n, sx, sy, incx = 1, incy = 1, c, s } = params;
  
  // Validate inputs
  if (n <= 0) return;
  if (incx === 0 || incy === 0) return;
  if (sx.length === 0 || sy.length === 0) return;
  
  const context = await getWebGPUContext();
  const device = context.getDevice();
  
  // Create copies of data for GPU processing
  const xData = new Float32Array(sx);
  const yData = new Float32Array(sy);
  const xBuffer = context.createBuffer(xData);
  const yBuffer = context.createBuffer(yData);
  
  // Create uniform buffer for parameters (16-byte aligned)
  const paramsData = new ArrayBuffer(16); // 3 u32s/i32s + padding for 16-byte alignment
  const paramsView = new DataView(paramsData);
  paramsView.setUint32(0, n, true);
  paramsView.setInt32(4, incx, true);
  paramsView.setInt32(8, incy, true);
  // Padding byte at offset 12 for alignment
  
  const paramsBuffer = device.createBuffer({
    size: paramsData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint8Array(paramsBuffer.getMappedRange()).set(new Uint8Array(paramsData));
  paramsBuffer.unmap();
  
  // Create uniform buffers for c and s (16-byte aligned each)
  const cData = new ArrayBuffer(16); // f32 + padding for 16-byte alignment
  const cView = new DataView(cData);
  cView.setFloat32(0, c, true);
  
  const cBuffer = device.createBuffer({
    size: cData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint8Array(cBuffer.getMappedRange()).set(new Uint8Array(cData));
  cBuffer.unmap();
  
  const sData = new ArrayBuffer(16); // f32 + padding for 16-byte alignment
  const sView = new DataView(sData);
  sView.setFloat32(0, s, true);
  
  const sBuffer = device.createBuffer({
    size: sData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint8Array(sBuffer.getMappedRange()).set(new Uint8Array(sData));
  sBuffer.unmap();
  
  // Create compute pipeline
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: device.createShaderModule({
        code: srotShader,
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
        resource: { buffer: cBuffer },
      },
      {
        binding: 4,
        resource: { buffer: sBuffer },
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
  
  // Read results
  const xResultData = await context.readBuffer(xBuffer, xData.byteLength);
  const yResultData = await context.readBuffer(yBuffer, yData.byteLength);
  const xResult = new Float32Array(xResultData);
  const yResult = new Float32Array(yResultData);
  
  // Cleanup
  xBuffer.destroy();
  yBuffer.destroy();
  paramsBuffer.destroy();
  cBuffer.destroy();
  sBuffer.destroy();
  
  // Copy results back to original arrays
  sx.set(xResult.slice(0, sx.length));
  sy.set(yResult.slice(0, sy.length));
}
