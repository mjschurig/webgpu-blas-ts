struct Params {
  nblocks: u32,
}

@group(0) @binding(0) var<storage, read> workspace: array<f64>;
@group(0) @binding(1) var<storage, read_write> result: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

const WORKGROUP_SIZE: u32 = 256u;

var<workgroup> shared_data: array<f64, WORKGROUP_SIZE>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  
  let tid = local_id.x;
  let nblocks = params.nblocks;
  
  // Load partial results into shared memory
  var sum = 0.0;
  if (tid < nblocks) {
    sum = workspace[tid];
  }
  shared_data[tid] = sum;
  workgroupBarrier();
  
  // Final reduction
  var step = WORKGROUP_SIZE / 2u;
  while (step > 0u) {
    if (tid < step) {
      shared_data[tid] += shared_data[tid + step];
    }
    workgroupBarrier();
    step /= 2u;
  }
  
  // Write final result
  if (tid == 0u) {
    result[0] = shared_data[0];
  }
}
