/**
 * SNRM2 Stage 2 Shader - Final reduction and square root for Euclidean norm
 * 
 * Combines the partial sums of squares from all workgroups and computes the final square root.
 * This stage runs only when multiple workgroups were used in stage 1.
 * This is the WebGPU implementation of the second stage of BLAS SNRM2 operation.
 */

struct Params {
  blocks: u32,
}

@group(0) @binding(0) var<storage, read> workspace: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

const WORKGROUP_SIZE: u32 = 256u;

var<workgroup> shared_data: array<f32, WORKGROUP_SIZE>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(global_invocation_id) global_id: vec3<u32>
) {
  let tid = local_id.x;
  let global_tid = global_id.x;
  let blocks = params.blocks;
  
  var sum: f32 = 0.0;
  
  // Each thread loads one or more partial results
  if (global_tid < blocks) {
    sum = workspace[global_tid];
  }
  
  // Also handle the case where blocks > WORKGROUP_SIZE
  // Each thread may need to accumulate multiple workspace values
  var idx = global_tid + WORKGROUP_SIZE;
  while (idx < blocks) {
    sum += workspace[idx];
    idx += WORKGROUP_SIZE;
  }
  
  // Store in shared memory for reduction
  shared_data[tid] = sum;
  workgroupBarrier();
  
  // Perform reduction within workgroup
  var stride = WORKGROUP_SIZE / 2u;
  while (stride > 0u) {
    if (tid < stride) {
      shared_data[tid] += shared_data[tid + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }
  
  // Compute final result with square root
  if (tid == 0u) {
    result[0] = sqrt(shared_data[0]);
  }
}
