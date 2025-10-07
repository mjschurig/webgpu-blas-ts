struct Params {
  n: u32,
  incx: i32,
  nblocks: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> workspace: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

// ROCm-style workgroup size and window size
const WORKGROUP_SIZE: u32 = 256u;
const WIN: u32 = 4u; // Elements per thread (WIN parameter from ROCm)

// Shared memory for workgroup reduction
var<workgroup> shared_data: array<f32, WORKGROUP_SIZE>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(workgroup_id) workgroup_id: vec3<u32>) {
  
  let tid = local_id.x;
  let n = params.n;
  let incx = params.incx;
  let nblocks = params.nblocks;
  
  // Each thread processes WIN elements starting from its global position
  var sum = 0.0f;
  let global_tid = global_id.x;
  let total_threads = WORKGROUP_SIZE * nblocks;
  
  // Process WIN elements per thread with proper stride
  for (var j = 0u; j < WIN; j++) {
    let element_id = global_tid + j * total_threads;
    if (element_id < n) {
      let idx = element_id * u32(incx);
      if (idx < arrayLength(&input)) {
        sum += abs(input[idx]);
      }
    }
  }
  
  // Store partial sum in shared memory
  shared_data[tid] = sum;
  workgroupBarrier();
  
  // Workgroup-level reduction (binary tree reduction)
  var step = WORKGROUP_SIZE / 2u;
  while (step > 0u) {
    if (tid < step) {
      shared_data[tid] += shared_data[tid + step];
    }
    workgroupBarrier();
    step /= 2u;
  }
  
  // First thread writes the workgroup result to workspace
  if (tid == 0u) {
    workspace[workgroup_id.x] = shared_data[0];
  }
}
