/**
 * SSCAL - Scale vector by scalar (single precision)
 * 
 * Performs the operation x := alpha * x where alpha is a scalar,
 * x is a vector. This is a WebGPU implementation of the reference BLAS SSCAL function.
 */

/// <reference types="@webgpu/types" />

import { getWebGPUContext } from '../../webgpu/context';
import sscalShader from '../../shaders/sscal.wgsl';

/**
 * SSCAL parameters
 */
export interface SSCALParams {
  /** Number of elements in the vector */
  n: number;
  /** Scalar multiplier alpha */
  alpha: number;
  /** Input/output vector x */
  sx: Float32Array;
  /** Increment/stride between elements in x (default: 1) */
  incx?: number;
}

/**
 * Scale vector x by scalar alpha using WebGPU (single precision)
 * Modifies the x vector in-place.
 */
export async function sscal(params: SSCALParams): Promise<void> {
  const { n, alpha, sx, incx = 1 } = params;
  
  // Validate inputs
  if (n <= 0) return;
  if (incx === 0) return;
  if (sx.length === 0) return;
  
  // Early return if alpha is 1.0 (no-op optimization)
  if (alpha === 1.0) return;
  
  const context = await getWebGPUContext();
  const device = context.getDevice();
  
  // Create a copy of x data for GPU processing
  const xData = new Float32Array(sx);
  const xBuffer = context.createBuffer(xData);
  
  // Create uniform buffer for parameters (16-byte aligned)
  const paramsData = new ArrayBuffer(16); // 2 u32s/i32s + padding for 16-byte alignment
  const paramsView = new DataView(paramsData);
  paramsView.setUint32(0, n, true);
  paramsView.setInt32(4, incx, true);
  // Padding bytes at offset 8-15 for alignment
  
  const paramsBuffer = device.createBuffer({
    size: paramsData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint8Array(paramsBuffer.getMappedRange()).set(new Uint8Array(paramsData));
  paramsBuffer.unmap();
  
  // Create uniform buffer for alpha (16-byte aligned)
  const alphaData = new ArrayBuffer(16); // f32 + padding for 16-byte alignment
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
        code: sscalShader,
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
        resource: { buffer: paramsBuffer },
      },
      {
        binding: 2,
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
  const resultData = await context.readBuffer(xBuffer, xData.byteLength);
  const result = new Float32Array(resultData);
  
  // Cleanup
  xBuffer.destroy();
  paramsBuffer.destroy();
  alphaBuffer.destroy();
  
  // Copy result back to original sx array
  sx.set(result.slice(0, sx.length));
}
