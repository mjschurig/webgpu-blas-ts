/**
 * ISAMIN Stage 1 Shader - Find local minimum absolute values and indices within workgroups
 * 
 * Computes partial minimum absolute values and their indices within each workgroup.
 * Each workgroup processes WORKGROUP_SIZE * WIN elements and outputs one min value/index pair.
 * This is the WebGPU implementation of the first stage of BLAS ISAMIN operation.
 */

struct IndexValue {
  index: u32,  // 1-based index (0 means invalid)
  value: f32,  // absolute value
}

struct Params {
  n: u32,
  incx: i32,
  blocks: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> workspace: array<IndexValue>;
@group(0) @binding(2) var<uniform> params: Params;

// Workgroup parameters matching ROCm configuration
const WORKGROUP_SIZE: u32 = 256u;
const WIN: u32 = 4u; // Work items (elements per thread)

var<workgroup> shared_data: array<IndexValue, WORKGROUP_SIZE>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
  let tid = local_id.x;
  let global_tid = global_id.x;
  let block_id = workgroup_id.x;
  
  let n = params.n;
  let incx = params.incx;
  
  var local_min = IndexValue(0u, 0.0f); // 0 index means invalid
  
  // Each thread processes WIN elements
  for (var i: u32 = 0u; i < WIN; i = i + 1u) {
    let idx = global_tid * WIN + i;
    if (idx < n) {
      // Calculate actual array index with stride
      let x_idx = u32(max(0, i32(idx) * incx));
      
      if (x_idx < arrayLength(&x)) {
        let abs_val = abs(x[x_idx]);
        let one_based_idx = idx + 1u; // Convert to 1-based index
        
        // Update local minimum
        if (local_min.index == 0u || abs_val < local_min.value || 
           (abs_val == local_min.value && one_based_idx < local_min.index)) {
          local_min.index = one_based_idx;
          local_min.value = abs_val;
        }
      }
    }
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
  
  // Write workgroup result to workspace
  if (tid == 0u) {
    workspace[block_id] = shared_data[0];
  }
}
