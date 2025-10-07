/**
 * SGEMV Transpose Shader - General matrix-vector multiply (transpose case)
 * 
 * Computes y = alpha * A^T * x + beta * y
 * where A is m x n, A^T is n x m, x is m x 1, y is n x 1
 * 
 * This is the WebGPU implementation of the BLAS SGEMV operation (transpose case).
 */

struct Params {
  m: u32,        // number of rows of matrix A
  n: u32,        // number of columns of matrix A
  lda: u32,      // leading dimension of A (stride between columns)
  incx: i32,     // increment between elements of x
  incy: i32,     // increment between elements of y
}

@group(0) @binding(0) var<storage, read> A: array<f32>;      // Matrix A (m x n)
@group(0) @binding(1) var<storage, read> x: array<f32>;      // Vector x (m x 1)
@group(0) @binding(2) var<storage, read_write> y: array<f32>; // Vector y (n x 1)
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<uniform> alpha: f32;
@group(0) @binding(5) var<uniform> beta: f32;

// Workgroup size optimized for matrix operations
const WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let col = global_id.x;
  let m = params.m;
  let n = params.n;
  let lda = params.lda;
  let incx = params.incx;
  let incy = params.incy;
  
  // Check bounds for output vector (n elements for transpose)
  if (col >= n) {
    return;
  }
  
  // Calculate y index with proper increment
  let y_idx = u32(max(0, i32(col) * incy));
  
  // Bounds checking for y array
  if (y_idx >= arrayLength(&y)) {
    return;
  }
  
  // Compute dot product of column 'col' of A with vector x
  var sum = 0.0;
  
  for (var row = 0u; row < m; row++) {
    // Calculate A index: A[row, col] = A[col * lda + row] (column-major)
    let a_idx = col * lda + row;
    
    // Calculate x index with proper increment
    let x_idx = u32(max(0, i32(row) * incx));
    
    // Bounds checking
    if (a_idx < arrayLength(&A) && x_idx < arrayLength(&x)) {
      sum += A[a_idx] * x[x_idx];
    }
  }
  
  // Apply alpha and beta: y = alpha * (A^T * x) + beta * y
  if (beta == 0.0) {
    y[y_idx] = alpha * sum;
  } else {
    y[y_idx] = alpha * sum + beta * y[y_idx];
  }
}
