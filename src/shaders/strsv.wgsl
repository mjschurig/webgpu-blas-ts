/**
 * STRSV Shader - Triangular solve
 * 
 * Solves the triangular system: A * x = b or A^T * x = b
 * where A is a triangular n x n matrix and x, b are n-element vectors.
 * 
 * The vector x is both input (b) and output (solution).
 * Only the upper ('U') or lower ('L') triangular part of A is referenced,
 * and the matrix can be unit ('U') or non-unit ('N') triangular.
 * 
 * This is the WebGPU implementation of the BLAS STRSV operation.
 * Note: This implementation uses a sequential approach suitable for GPU.
 */

struct Params {
  n: u32,        // dimension of matrix A (n x n)
  lda: u32,      // leading dimension of A (stride between columns)
  incx: i32,     // increment between elements of x
  uplo: u32,     // 0 = lower ('L'), 1 = upper ('U')
  trans: u32,    // 0 = no transpose ('N'), 1 = transpose ('T')
  diag: u32,     // 0 = non-unit ('N'), 1 = unit ('U')
}

@group(0) @binding(0) var<storage, read> A: array<f32>;           // Triangular matrix A (n x n)
@group(0) @binding(1) var<storage, read> x_in: array<f32>;        // Input vector b
@group(0) @binding(2) var<storage, read_write> x_out: array<f32>; // Output vector x (solution)
@group(0) @binding(3) var<uniform> params: Params;

// Use single thread for sequential solve
const WORKGROUP_SIZE: u32 = 1u;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // Only process with first thread
  if (global_id.x != 0u) {
    return;
  }
  
  let n = params.n;
  let lda = params.lda;
  let incx = params.incx;
  let uplo = params.uplo;
  let trans = params.trans;
  let diag = params.diag;
  
  // Copy input to output initially
  for (var i = 0u; i < n; i++) {
    let x_idx = u32(max(0, i32(i) * incx));
    if (x_idx < arrayLength(&x_in) && x_idx < arrayLength(&x_out)) {
      x_out[x_idx] = x_in[x_idx];
    }
  }
  
  // Solve triangular system
  if (trans == 0u) { // No transpose: A * x = b
    if (uplo == 1u) { // Upper triangular - solve backward
      for (var i_rev = 0u; i_rev < n; i_rev++) {
        let i = n - 1u - i_rev;
        let x_i_idx = u32(max(0, i32(i) * incx));
        
        if (x_i_idx >= arrayLength(&x_out)) {
          continue;
        }
        
        var sum = 0.0;
        
        // Subtract A[i,j] * x[j] for j > i
        for (var j = i + 1u; j < n; j++) {
          let x_j_idx = u32(max(0, i32(j) * incx));
          let a_idx = j * lda + i;
          
          if (x_j_idx < arrayLength(&x_out) && a_idx < arrayLength(&A)) {
            sum += A[a_idx] * x_out[x_j_idx];
          }
        }
        
        // Solve for x[i]
        var result = x_out[x_i_idx] - sum;
        
        if (diag == 0u) { // Non-unit diagonal
          let a_ii_idx = i * lda + i;
          if (a_ii_idx < arrayLength(&A) && A[a_ii_idx] != 0.0) {
            result = result / A[a_ii_idx];
          }
        }
        // For unit diagonal, x[i] = result (no division needed)
        
        x_out[x_i_idx] = result;
      }
    } else { // Lower triangular - solve forward
      for (var i = 0u; i < n; i++) {
        let x_i_idx = u32(max(0, i32(i) * incx));
        
        if (x_i_idx >= arrayLength(&x_out)) {
          continue;
        }
        
        var sum = 0.0;
        
        // Subtract A[i,j] * x[j] for j < i
        for (var j = 0u; j < i; j++) {
          let x_j_idx = u32(max(0, i32(j) * incx));
          let a_idx = j * lda + i;
          
          if (x_j_idx < arrayLength(&x_out) && a_idx < arrayLength(&A)) {
            sum += A[a_idx] * x_out[x_j_idx];
          }
        }
        
        // Solve for x[i]
        var result = x_out[x_i_idx] - sum;
        
        if (diag == 0u) { // Non-unit diagonal
          let a_ii_idx = i * lda + i;
          if (a_ii_idx < arrayLength(&A) && A[a_ii_idx] != 0.0) {
            result = result / A[a_ii_idx];
          }
        }
        // For unit diagonal, x[i] = result (no division needed)
        
        x_out[x_i_idx] = result;
      }
    }
  } else { // Transpose: A^T * x = b
    if (uplo == 1u) { // Upper triangular becomes lower after transpose - solve forward
      for (var i = 0u; i < n; i++) {
        let x_i_idx = u32(max(0, i32(i) * incx));
        
        if (x_i_idx >= arrayLength(&x_out)) {
          continue;
        }
        
        var sum = 0.0;
        
        // Subtract A^T[i,j] * x[j] = A[j,i] * x[j] for j < i
        for (var j = 0u; j < i; j++) {
          let x_j_idx = u32(max(0, i32(j) * incx));
          let a_idx = i * lda + j; // A[j,i] in column-major
          
          if (x_j_idx < arrayLength(&x_out) && a_idx < arrayLength(&A)) {
            sum += A[a_idx] * x_out[x_j_idx];
          }
        }
        
        // Solve for x[i]
        var result = x_out[x_i_idx] - sum;
        
        if (diag == 0u) { // Non-unit diagonal
          let a_ii_idx = i * lda + i;
          if (a_ii_idx < arrayLength(&A) && A[a_ii_idx] != 0.0) {
            result = result / A[a_ii_idx];
          }
        }
        
        x_out[x_i_idx] = result;
      }
    } else { // Lower triangular becomes upper after transpose - solve backward
      for (var i_rev = 0u; i_rev < n; i_rev++) {
        let i = n - 1u - i_rev;
        let x_i_idx = u32(max(0, i32(i) * incx));
        
        if (x_i_idx >= arrayLength(&x_out)) {
          continue;
        }
        
        var sum = 0.0;
        
        // Subtract A^T[i,j] * x[j] = A[j,i] * x[j] for j > i
        for (var j = i + 1u; j < n; j++) {
          let x_j_idx = u32(max(0, i32(j) * incx));
          let a_idx = i * lda + j; // A[j,i] in column-major
          
          if (x_j_idx < arrayLength(&x_out) && a_idx < arrayLength(&A)) {
            sum += A[a_idx] * x_out[x_j_idx];
          }
        }
        
        // Solve for x[i]
        var result = x_out[x_i_idx] - sum;
        
        if (diag == 0u) { // Non-unit diagonal
          let a_ii_idx = i * lda + i;
          if (a_ii_idx < arrayLength(&A) && A[a_ii_idx] != 0.0) {
            result = result / A[a_ii_idx];
          }
        }
        
        x_out[x_i_idx] = result;
      }
    }
  }
}
