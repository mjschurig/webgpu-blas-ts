/**
 * ISAMIN - Index of minimum absolute value (single precision)
 * 
 * Finds the index of the element with the smallest absolute value in a vector.
 * Returns a 1-based index (FORTRAN convention).
 * This is a WebGPU implementation of the reference BLAS ISAMIN function.
 */

/// <reference types="@webgpu/types" />

import { getWebGPUContext } from '../../webgpu/context';
import isaminStage1Shader from '../../shaders/isamin_stage1.wgsl';
import isaminStage2Shader from '../../shaders/isamin_stage2.wgsl';

/**
 * ISAMIN parameters
 */
export interface ISAMINParams {
  /** Number of elements in the vector */
  n: number;
  /** Input vector x */
  sx: Float32Array;
  /** Increment/stride between elements (default: 1) */
  incx?: number;
}

/**
 * Find the index of the minimum absolute value in a vector using WebGPU (single precision)
 * Returns 1-based index (FORTRAN convention), 0 if n <= 0
 */
export async function isamin(params: ISAMINParams): Promise<number> {
  const { n, sx, incx = 1 } = params;
  
  // Validate inputs
  if (n <= 0) return 0;
  if (incx <= 0) return 0;
  if (sx.length === 0) return 0;
  
  const context = await getWebGPUContext();
  const device = context.getDevice();
  
  // ROCm-style parameters
  const WORKGROUP_SIZE = 256;
  const WIN = 4; // Elements per thread
  const blocks = Math.ceil(n / (WORKGROUP_SIZE * WIN));
  
  // Create buffers
  const inputBuffer = context.createBuffer(sx);
  // IndexValue struct size: 8 bytes (u32 + f32)
  const workspaceSize = blocks * 8; // One IndexValue per block
  const workspaceBuffer = context.createOutputBuffer(workspaceSize);
  const resultBuffer = context.createOutputBuffer(4); // Single u32 result
  
  // Stage 1: Create uniform buffer for stage 1 parameters (16-byte aligned)
  const stage1ParamsData = new ArrayBuffer(16); // 3 u32s + padding for 16-byte alignment
  const stage1ParamsView = new DataView(stage1ParamsData);
  stage1ParamsView.setUint32(0, n, true);
  stage1ParamsView.setInt32(4, incx, true);
  stage1ParamsView.setUint32(8, blocks, true);
  // Padding byte at offset 12 for alignment
  
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
        code: isaminStage1Shader,
      }),
      entryPoint: 'main',
    },
  });
  
  const stage2Pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: device.createShaderModule({
        code: isaminStage2Shader,
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
        resource: { buffer: inputBuffer },
      },
      {
        binding: 1,
        resource: { buffer: workspaceBuffer },
      },
      {
        binding: 2,
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
  
  // Stage 1: Parallel reduction within workgroups to find local minimums
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
  } else {
    // If only one block, we need to extract the index from the workspace
    // For simplicity, we'll read the workspace and extract the index on CPU
    // In a more optimized version, we'd have a single-block shader variant
  }
  
  device.queue.submit([commandEncoder.finish()]);
  
  // Read final result
  let minIndex: number;
  if (blocks > 1) {
    const resultData = await context.readBuffer(resultBuffer, 4);
    const resultView = new Uint32Array(resultData);
    minIndex = resultView[0] || 0;
  } else {
    // Single block case - need to extract index from IndexValue struct
    const resultData = await context.readBuffer(workspaceBuffer, 8);
    const resultView = new Uint32Array(resultData);
    minIndex = resultView[0] || 0; // First u32 is the index
  }
  
  // Cleanup
  inputBuffer.destroy();
  workspaceBuffer.destroy();
  resultBuffer.destroy();
  stage1ParamsBuffer.destroy();
  stage2ParamsBuffer.destroy();
  
  return minIndex;
}
