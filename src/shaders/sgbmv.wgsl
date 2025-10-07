/**
 * SGBMV Shader - General banded matrix-vector multiply
 * 
 * Computes y = alpha * A * x + beta * y or y = alpha * A^T * x + beta * y
 * where A is an m x n banded matrix with kl sub-diagonals and ku super-diagonals,
 * x and y are vectors, and alpha and beta are scalars.
 * 
 * This is the WebGPU implementation of the BLAS SGBMV operation.
 */

struct Params {
  m: u32,        // number of rows of matrix A
  n: u32,        // number of columns of matrix A
  kl: u32,       // number of sub-diagonals of A
  ku: u32,       // number of super-diagonals of A
  lda: u32,      // leading dimension of A (must be >= kl + ku + 1)
  incx: i32,     // increment between elements of x
  incy: i32,     // increment between elements of y
  trans: u32,    // 0 = no transpose ('N'), 1 = transpose ('T')
}

@group(0) @binding(0) var<storage, read> A: array<f32>;       // Banded matrix A (banded storage)
@group(0) @binding(1) var<storage, read> x: array<f32>;       // Vector x
@group(0) @binding(2) var<storage, read_write> y: array<f32>; // Vector y
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<uniform> alpha: f32;
@group(0) @binding(5) var<uniform> beta: f32;

// Workgroup size optimized for matrix operations
const WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let thread_id = global_id.x;
  let m = params.m;
  let n = params.n;
  let kl = params.kl;
  let ku = params.ku;
  let lda = params.lda;
  let incx = params.incx;
  let incy = params.incy;
  let trans = params.trans;
  
  if (trans == 0u) { // No transpose: y = alpha * A * x + beta * y
    let row = thread_id;
    if (row >= m) {
      return;
    }
    
    // Calculate y index with proper increment
    let y_idx = u32(max(0, i32(row) * incy));
    
    // Bounds checking for y array
    if (y_idx >= arrayLength(&y)) {
      return;
    }
    
    // Compute dot product of row 'row' with vector x (banded matrix)
    var sum = 0.0;
    
    // For banded storage: A[i,j] is stored at A[(ku + i - j) + j * lda]
    // where ku + i - j >= 0 and ku + i - j < lda (within band)
    let i = row;
    let j_start = max(0, i32(i) - i32(kl));
    let j_end = min(i32(n), i32(i) + i32(ku) + 1);
    
    for (var j_signed = j_start; j_signed < j_end; j_signed++) {
      let j = u32(j_signed);
      
      // Calculate banded storage index
      let band_idx = ku + i - j;
      let a_idx = band_idx + j * lda;
      
      // Calculate x index with proper increment
      let x_idx = u32(max(0, i32(j) * incx));
      
      // Bounds checking
      if (a_idx < arrayLength(&A) && x_idx < arrayLength(&x)) {
        sum += A[a_idx] * x[x_idx];
      }
    }
    
    // Apply alpha and beta: y = alpha * (A * x) + beta * y
    if (beta == 0.0) {
      y[y_idx] = alpha * sum;
    } else {
      y[y_idx] = alpha * sum + beta * y[y_idx];
    }
  } else { // Transpose: y = alpha * A^T * x + beta * y
    let col = thread_id;
    if (col >= n) {
      return;
    }
    
    // Calculate y index with proper increment
    let y_idx = u32(max(0, i32(col) * incy));
    
    // Bounds checking for y array
    if (y_idx >= arrayLength(&y)) {
      return;
    }
    
    // Compute dot product of column 'col' with vector x (banded matrix transpose)
    var sum = 0.0;
    
    // For transpose, we need A^T[col, row] = A[row, col]
    let j = col;
    let i_start = max(0, i32(j) - i32(ku));
    let i_end = min(i32(m), i32(j) + i32(kl) + 1);
    
    for (var i_signed = i_start; i_signed < i_end; i_signed++) {
      let i = u32(i_signed);
      
      // Calculate banded storage index for A[i,j]
      let band_idx = ku + i - j;
      let a_idx = band_idx + j * lda;
      
      // Calculate x index with proper increment
      let x_idx = u32(max(0, i32(i) * incx));
      
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
}
