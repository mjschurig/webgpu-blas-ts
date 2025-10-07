/**
 * SSYMV Shader - Symmetric matrix-vector multiply
 * 
 * Computes y = alpha * A * x + beta * y
 * where A is a symmetric n x n matrix, x and y are n-element vectors,
 * and alpha and beta are scalars.
 * 
 * Only the upper ('U') or lower ('L') triangular part of A is referenced.
 * 
 * This is the WebGPU implementation of the BLAS SSYMV operation.
 */

struct Params {
  n: u32,        // dimension of matrix A (n x n)
  lda: u32,      // leading dimension of A (stride between columns)
  incx: i32,     // increment between elements of x
  incy: i32,     // increment between elements of y
  uplo: u32,     // 0 = lower ('L'), 1 = upper ('U')
}

@group(0) @binding(0) var<storage, read> A: array<f32>;       // Symmetric matrix A (n x n)
@group(0) @binding(1) var<storage, read> x: array<f32>;       // Vector x (n x 1)
@group(0) @binding(2) var<storage, read_write> y: array<f32>; // Vector y (n x 1)
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<uniform> alpha: f32;
@group(0) @binding(5) var<uniform> beta: f32;

// Workgroup size optimized for matrix operations
const WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let row = global_id.x;
  let n = params.n;
  let lda = params.lda;
  let incx = params.incx;
  let incy = params.incy;
  let uplo = params.uplo;
  
  // Check bounds for output vector
  if (row >= n) {
    return;
  }
  
  // Calculate y index with proper increment
  let y_idx = u32(max(0, i32(row) * incy));
  
  // Bounds checking for y array
  if (y_idx >= arrayLength(&y)) {
    return;
  }
  
  // Compute symmetric matrix-vector product
  var sum = 0.0;
  
  for (var col = 0u; col < n; col++) {
    // Calculate x index with proper increment
    let x_idx = u32(max(0, i32(col) * incx));
    
    // Bounds checking for x
    if (x_idx >= arrayLength(&x)) {
      continue;
    }
    
    var a_val: f32;
    
    if (uplo == 1u) { // Upper triangular
      if (row <= col) {
        // Use upper triangular part: A[row, col]
        let a_idx = col * lda + row;
        if (a_idx < arrayLength(&A)) {
          a_val = A[a_idx];
        } else {
          continue;
        }
      } else {
        // Use symmetry: A[row, col] = A[col, row]
        let a_idx = row * lda + col;
        if (a_idx < arrayLength(&A)) {
          a_val = A[a_idx];
        } else {
          continue;
        }
      }
    } else { // Lower triangular
      if (row >= col) {
        // Use lower triangular part: A[row, col]
        let a_idx = col * lda + row;
        if (a_idx < arrayLength(&A)) {
          a_val = A[a_idx];
        } else {
          continue;
        }
      } else {
        // Use symmetry: A[row, col] = A[col, row]
        let a_idx = row * lda + col;
        if (a_idx < arrayLength(&A)) {
          a_val = A[a_idx];
        } else {
          continue;
        }
      }
    }
    
    sum += a_val * x[x_idx];
  }
  
  // Apply alpha and beta: y = alpha * (A * x) + beta * y
  if (beta == 0.0) {
    y[y_idx] = alpha * sum;
  } else {
    y[y_idx] = alpha * sum + beta * y[y_idx];
  }
}
