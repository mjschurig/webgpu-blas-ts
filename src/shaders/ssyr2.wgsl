/**
 * SSYR2 Shader - Symmetric rank-2 update
 * 
 * Performs the operation: A := alpha * (x * y^T + y * x^T) + A
 * where A is a symmetric n x n matrix, x and y are n-element vectors,
 * and alpha is a scalar.
 * 
 * Only the upper ('U') or lower ('L') triangular part of A is modified.
 * 
 * This is the WebGPU implementation of the BLAS SSYR2 operation.
 */

struct Params {
  n: u32,        // dimension of matrix A (n x n)
  lda: u32,      // leading dimension of A (stride between columns)
  incx: i32,     // increment between elements of x
  incy: i32,     // increment between elements of y
  uplo: u32,     // 0 = lower ('L'), 1 = upper ('U')
}

@group(0) @binding(0) var<storage, read_write> A: array<f32>; // Symmetric matrix A (n x n)
@group(0) @binding(1) var<storage, read> x: array<f32>;       // Vector x (n x 1)
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
  
  let n = params.n;
  let lda = params.lda;
  let incx = params.incx;
  let incy = params.incy;
  let uplo = params.uplo;
  
  // Check bounds
  if (row >= n || col >= n) {
    return;
  }
  
  // Only update the specified triangular part
  var should_update = false;
  if (uplo == 1u) { // Upper triangular
    should_update = (row <= col);
  } else { // Lower triangular
    should_update = (row >= col);
  }
  
  if (!should_update) {
    return;
  }
  
  // Calculate matrix index: A[i, j] = A[j * lda + i] (column-major)
  let a_idx = col * lda + row;
  
  // Calculate vector indices with proper increments
  let x_row_idx = u32(max(0, i32(row) * incx));
  let x_col_idx = u32(max(0, i32(col) * incx));
  let y_row_idx = u32(max(0, i32(row) * incy));
  let y_col_idx = u32(max(0, i32(col) * incy));
  
  // Bounds checking for all arrays
  if (a_idx < arrayLength(&A) && 
      x_row_idx < arrayLength(&x) && 
      x_col_idx < arrayLength(&x) &&
      y_row_idx < arrayLength(&y) && 
      y_col_idx < arrayLength(&y)) {
    
    // Perform symmetric rank-2 update: A[i,j] = alpha * (x[i]*y[j] + y[i]*x[j]) + A[i,j]
    let x_row_val = x[x_row_idx];
    let x_col_val = x[x_col_idx];
    let y_row_val = y[y_row_idx];
    let y_col_val = y[y_col_idx];
    
    let update = alpha * (x_row_val * y_col_val + y_row_val * x_col_val);
    
    A[a_idx] = A[a_idx] + update;
  }
}
