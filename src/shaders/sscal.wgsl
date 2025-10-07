/**
 * SSCAL Shader - Scale vector by scalar
 * 
 * Performs the operation: x[i*incx] = alpha * x[i*incx] for i = 0..n-1
 * This is the WebGPU implementation of the BLAS SSCAL operation.
 */

struct Params {
  n: u32,
  incx: i32,
}

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<uniform> alpha: f32;

// Workgroup size following the pattern from existing shaders
const WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let tid = global_id.x;
  let n = params.n;
  let incx = params.incx;
  
  // Check bounds
  if (tid >= n) {
    return;
  }
  
  // Early return if alpha is 1.0 (no-op optimization)
  if (alpha == 1.0) {
    return;
  }
  
  // Calculate index with proper increment
  let x_idx = u32(max(0, i32(tid) * incx));
  
  // Bounds checking for array
  if (x_idx < arrayLength(&x)) {
    // Perform SCAL operation: x = alpha * x
    x[x_idx] = alpha * x[x_idx];
  }
}
