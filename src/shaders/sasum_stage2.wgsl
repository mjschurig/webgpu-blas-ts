struct Params {
  nblocks: u32,
}

@group(0) @binding(0) var<storage, read> workspace: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

// Workgroup size for final reduction
const WORKGROUP_SIZE: u32 = 256u;

// Shared memory for final reduction
var<workgroup> shared_data: array<f32, WORKGROUP_SIZE>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  
  let tid = local_id.x;
  let nblocks = params.nblocks;
  
  // Load data from workspace into shared memory
  var sum = 0.0f;
  if (tid < nblocks) {
    sum = workspace[tid];
  }
  
  shared_data[tid] = sum;
  workgroupBarrier();
  
  // Binary tree reduction
  var step = WORKGROUP_SIZE / 2u;
  while (step > 0u) {
    if (tid < step && tid + step < nblocks) {
      shared_data[tid] += shared_data[tid + step];
    }
    workgroupBarrier();
    step /= 2u;
  }
  
  // First thread writes final result
  if (tid == 0u) {
    result[0] = shared_data[0];
  }
}
