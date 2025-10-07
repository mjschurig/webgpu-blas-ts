/**
 * SGER Shader - General rank-1 update
 * 
 * Performs the operation: A := alpha * x * y^T + A
 * where A is an m x n matrix, x is an m-element vector, y is an n-element vector,
 * and alpha is a scalar.
 * 
 * This is the WebGPU implementation of the BLAS SGER operation.
 */

struct Params {
  m: u32,        // number of rows of matrix A
  n: u32,        // number of columns of matrix A
  lda: u32,      // leading dimension of A (stride between columns)
  incx: i32,     // increment between elements of x
  incy: i32,     // increment between elements of y
}

@group(0) @binding(0) var<storage, read_write> A: array<f32>; // Matrix A (m x n)
@group(0) @binding(1) var<storage, read> x: array<f32>;       // Vector x (m x 1)
@group(0) @binding(2) var<storage, read> y: array<f32>;       // Vector y (n x 1)
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<uniform> alpha: f32;

// Workgroup size for 2D dispatch (8x8 threads per workgroup)
const WORKGROUP_SIZE_X: u32 = 8u;
const WORKGROUP_SIZE_Y: u32 = 8u;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let row = global_id.y;  // Row index (i)
  let col = global_id.x;  // Column index (j)
  
  let m = params.m;
  let n = params.n;
  let lda = params.lda;
  let incx = params.incx;
  let incy = params.incy;
  
  // Check bounds
  if (row >= m || col >= n) {
    return;
  }
  
  // Calculate matrix index: A[i, j] = A[j * lda + i] (column-major)
  let a_idx = col * lda + row;
  
  // Calculate vector indices with proper increments
  let x_idx = u32(max(0, i32(row) * incx));
  let y_idx = u32(max(0, i32(col) * incy));
  
  // Bounds checking for all arrays
  if (a_idx < arrayLength(&A) && 
      x_idx < arrayLength(&x) && 
      y_idx < arrayLength(&y)) {
    
    // Perform rank-1 update: A[i,j] = alpha * x[i] * y[j] + A[i,j]
    let x_val = x[x_idx];
    let y_val = y[y_idx];
    let update = alpha * x_val * y_val;
    
    A[a_idx] = A[a_idx] + update;
  }
}
