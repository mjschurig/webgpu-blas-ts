/**
 * ISAMIN Stage 2 Shader - Final reduction to find global minimum
 * 
 * Combines the partial minimum values and indices from all workgroups into a single final result.
 * This stage runs only when multiple workgroups were used in stage 1.
 * This is the WebGPU implementation of the second stage of BLAS ISAMIN operation.
 */

struct IndexValue {
  index: u32,  // 1-based index (0 means invalid)
  value: f32,  // absolute value
}

struct Params {
  blocks: u32,
}

@group(0) @binding(0) var<storage, read> workspace: array<IndexValue>;
@group(0) @binding(1) var<storage, read_write> result: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

const WORKGROUP_SIZE: u32 = 256u;

var<workgroup> shared_data: array<IndexValue, WORKGROUP_SIZE>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(global_invocation_id) global_id: vec3<u32>
) {
  let tid = local_id.x;
  let global_tid = global_id.x;
  let blocks = params.blocks;
  
  var local_min = IndexValue(0u, 0.0f);
  
  // Each thread loads one or more partial results
  if (global_tid < blocks) {
    local_min = workspace[global_tid];
  }
  
  // Also handle the case where blocks > WORKGROUP_SIZE
  // Each thread may need to process multiple workspace values
  var idx = global_tid + WORKGROUP_SIZE;
  while (idx < blocks) {
    let other = workspace[idx];
    // Compare and update minimum
    if (other.index != 0u && (local_min.index == 0u || 
        other.value < local_min.value ||
        (other.value == local_min.value && other.index < local_min.index))) {
      local_min = other;
    }
    idx += WORKGROUP_SIZE;
  }
  
  // Store in shared memory for reduction
  shared_data[tid] = local_min;
  workgroupBarrier();
  
  // Perform reduction within workgroup
  var stride = WORKGROUP_SIZE / 2u;
  while (stride > 0u) {
    if (tid < stride) {
      let other = shared_data[tid + stride];
      // Compare and update minimum
      if (other.index != 0u && (shared_data[tid].index == 0u || 
          other.value < shared_data[tid].value ||
          (other.value == shared_data[tid].value && other.index < shared_data[tid].index))) {
        shared_data[tid] = other;
      }
    }
    workgroupBarrier();
    stride = stride / 2u;
  }
  
  // Write final result (just the index)
  if (tid == 0u) {
    result[0] = shared_data[0].index;
  }
}
