/**
 * SDOT - Compute dot product of two vectors (single precision)
 * 
 * Computes the dot product of vectors x and y: sum(x[i] * y[i]).
 * This is a WebGPU implementation of the reference BLAS SDOT function.
 */

/// <reference types="@webgpu/types" />

import { getWebGPUContext } from '../../webgpu/context';
import sdotStage1Shader from '../../shaders/sdot_stage1.wgsl';
import sdotStage2Shader from '../../shaders/sdot_stage2.wgsl';

/**
 * SDOT parameters
 */
export interface SDOTParams {
  /** Number of elements in the vectors */
  n: number;
  /** First input vector x */
  sx: Float32Array;
  /** Increment/stride between elements in x (default: 1) */
  incx?: number;
  /** Second input vector y */
  sy: Float32Array;
  /** Increment/stride between elements in y (default: 1) */
  incy?: number;
}

/**
 * Compute the dot product of two vectors using WebGPU (single precision)
 */
export async function sdot(params: SDOTParams): Promise<number> {
  const { n, sx, sy, incx = 1, incy = 1 } = params;
  
  // Validate inputs
  if (n <= 0) return 0.0;
  if (incx <= 0 || incy <= 0) return 0.0;
  if (sx.length === 0 || sy.length === 0) return 0.0;
  
  const context = await getWebGPUContext();
  const device = context.getDevice();
  
  // ROCm-style parameters
  const WORKGROUP_SIZE = 256;
  const WIN = 4; // Elements per thread
  const blocks = Math.ceil(n / (WORKGROUP_SIZE * WIN));
  
  // Create buffers
  const xBuffer = context.createBuffer(sx);
  const yBuffer = context.createBuffer(sy);
  const workspaceSize = blocks * 4; // One f32 per block
  const workspaceBuffer = context.createOutputBuffer(workspaceSize);
  const resultBuffer = context.createOutputBuffer(4); // Single f32 result
  
  // Stage 1: Create uniform buffer for stage 1 parameters (16-byte aligned)
  const stage1ParamsData = new ArrayBuffer(16); // 4 u32s for 16-byte alignment
  const stage1ParamsView = new DataView(stage1ParamsData);
  stage1ParamsView.setUint32(0, n, true);
  stage1ParamsView.setInt32(4, incx, true);
  stage1ParamsView.setInt32(8, incy, true);
  stage1ParamsView.setUint32(12, blocks, true);
  
  const stage1ParamsBuffer = device.createBuffer({
    size: stage1ParamsData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint8Array(stage1ParamsBuffer.getMappedRange()).set(new Uint8Array(stage1ParamsData));
  stage1ParamsBuffer.unmap();
  
  // Stage 2: Create uniform buffer for stage 2 parameters (16-byte aligned)
  const stage2ParamsData = new ArrayBuffer(16); // 1 u32 + padding for 16-byte alignment
  const stage2ParamsView = new DataView(stage2ParamsData);
  stage2ParamsView.setUint32(0, blocks, true);
  
  const stage2ParamsBuffer = device.createBuffer({
    size: stage2ParamsData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint8Array(stage2ParamsBuffer.getMappedRange()).set(new Uint8Array(stage2ParamsData));
  stage2ParamsBuffer.unmap();
  
  // Create compute pipelines using imported WGSL shaders
  const stage1Pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: device.createShaderModule({
        code: sdotStage1Shader,
      }),
      entryPoint: 'main',
    },
  });
  
  const stage2Pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: device.createShaderModule({
        code: sdotStage2Shader,
      }),
      entryPoint: 'main',
    },
  });
  
  // Create bind groups
  const stage1BindGroup = device.createBindGroup({
    layout: stage1Pipeline.getBindGroupLayout(0),
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
        resource: { buffer: workspaceBuffer },
      },
      {
        binding: 3,
        resource: { buffer: stage1ParamsBuffer },
      },
    ],
  });
  
  const stage2BindGroup = device.createBindGroup({
    layout: stage2Pipeline.getBindGroupLayout(0),
    entries: [
      {
        binding: 0,
        resource: { buffer: workspaceBuffer },
      },
      {
        binding: 1,
        resource: { buffer: resultBuffer },
      },
      {
        binding: 2,
        resource: { buffer: stage2ParamsBuffer },
      },
    ],
  });
  
  // Execute two-stage reduction
  const commandEncoder = device.createCommandEncoder();
  
  // Stage 1: Parallel reduction within workgroups
  const stage1Pass = commandEncoder.beginComputePass();
  stage1Pass.setPipeline(stage1Pipeline);
  stage1Pass.setBindGroup(0, stage1BindGroup);
  stage1Pass.dispatchWorkgroups(blocks);
  stage1Pass.end();
  
  // Stage 2: Final reduction (only if we have multiple blocks)
  if (blocks > 1) {
    const stage2Pass = commandEncoder.beginComputePass();
    stage2Pass.setPipeline(stage2Pipeline);
    stage2Pass.setBindGroup(0, stage2BindGroup);
    stage2Pass.dispatchWorkgroups(1); // Single workgroup for final reduction
    stage2Pass.end();
  }
  
  device.queue.submit([commandEncoder.finish()]);
  
  // Read final result
  const resultData = await context.readBuffer(
    blocks > 1 ? resultBuffer : workspaceBuffer,
    4
  );
  const resultView = new Float32Array(resultData);
  const dotProduct = resultView[0] || 0.0;
  
  // Cleanup
  xBuffer.destroy();
  yBuffer.destroy();
  workspaceBuffer.destroy();
  resultBuffer.destroy();
  stage1ParamsBuffer.destroy();
  stage2ParamsBuffer.destroy();
  
  return dotProduct;
}
