struct Params {
  n: u32,
  incx: i32,
  incy: i32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<uniform> alpha: f32;

// Workgroup size following the pattern from dasum
const WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let tid = global_id.x;
  let n = params.n;
  let incx = params.incx;
  let incy = params.incy;
  
  // Check bounds
  if (tid >= n) {
    return;
  }
  
  // Calculate indices with proper increments
  let x_idx = u32(max(0, i32(tid) * incx));
  let y_idx = u32(max(0, i32(tid) * incy));
  
  // Bounds checking for arrays
  if (x_idx < arrayLength(&x) && y_idx < arrayLength(&y)) {
    // Perform AXPY operation: y = alpha * x + y
    y[y_idx] = alpha * x[x_idx] + y[y_idx];
  }
}
