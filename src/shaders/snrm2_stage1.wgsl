/**
 * SNRM2 Stage 1 Shader - Parallel reduction within workgroups for Euclidean norm
 * 
 * Computes partial sums of squares within each workgroup.
 * Each workgroup processes WORKGROUP_SIZE * WIN elements and outputs one sum.
 * This is the WebGPU implementation of the first stage of BLAS SNRM2 operation.
 */

struct Params {
  n: u32,
  incx: i32,
  blocks: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> workspace: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

// Workgroup parameters matching ROCm configuration
const WORKGROUP_SIZE: u32 = 256u;
const WIN: u32 = 4u; // Work items (elements per thread)

var<workgroup> shared_data: array<f32, WORKGROUP_SIZE>;

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
  
  var sum: f32 = 0.0;
  
  // Each thread processes WIN elements
  for (var i: u32 = 0u; i < WIN; i = i + 1u) {
    let idx = global_tid * WIN + i;
    if (idx < n) {
      // Calculate actual array index with stride
      let x_idx = u32(max(0, i32(idx) * incx));
      
      if (x_idx < arrayLength(&x)) {
        // Compute sum of squares: sum += x[i] * x[i]
        let val = x[x_idx];
        sum += val * val;
      }
    }
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
  
  // Write workgroup result to workspace
  if (tid == 0u) {
    workspace[block_id] = shared_data[0];
  }
}
