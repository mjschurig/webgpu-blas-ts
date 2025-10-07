/**
 * SROT Shader - Apply plane rotation to vectors
 * 
 * Applies a plane rotation transformation to vectors x and y:
 * temp_x = c * x + s * y
 * temp_y = c * y - s * x
 * x = temp_x, y = temp_y
 * 
 * This is the WebGPU implementation of the BLAS SROT operation.
 */

struct Params {
  n: u32,
  incx: i32,
  incy: i32,
}

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<uniform> c: f32;  // cosine
@group(0) @binding(4) var<uniform> s: f32;  // sine

// Workgroup size following the pattern from existing shaders
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
    // Apply plane rotation transformation
    let x_val = x[x_idx];
    let y_val = y[y_idx];
    
    let temp_x = c * x_val + s * y_val;
    let temp_y = c * y_val - s * x_val;
    
    x[x_idx] = temp_x;
    y[y_idx] = temp_y;
  }
}
