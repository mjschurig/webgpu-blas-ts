/**
 * STRMV Shader - Triangular matrix-vector multiply
 * 
 * Computes x := A * x or x := A^T * x
 * where A is a triangular n x n matrix and x is an n-element vector.
 * 
 * Only the upper ('U') or lower ('L') triangular part of A is referenced,
 * and the matrix can be unit ('U') or non-unit ('N') triangular.
 * 
 * This is the WebGPU implementation of the BLAS STRMV operation.
 */

struct Params {
  n: u32,        // dimension of matrix A (n x n)
  lda: u32,      // leading dimension of A (stride between columns)
  incx: i32,     // increment between elements of x
  uplo: u32,     // 0 = lower ('L'), 1 = upper ('U')
  trans: u32,    // 0 = no transpose ('N'), 1 = transpose ('T')
  diag: u32,     // 0 = non-unit ('N'), 1 = unit ('U')
}

@group(0) @binding(0) var<storage, read> A: array<f32>;       // Triangular matrix A (n x n)
@group(0) @binding(1) var<storage, read> x_in: array<f32>;    // Input vector x
@group(0) @binding(2) var<storage, read_write> x_out: array<f32>; // Output vector x
@group(0) @binding(3) var<uniform> params: Params;

// Workgroup size optimized for matrix operations
const WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let row = global_id.x;
  let n = params.n;
  let lda = params.lda;
  let incx = params.incx;
  let uplo = params.uplo;
  let trans = params.trans;
  let diag = params.diag;
  
  // Check bounds for output vector
  if (row >= n) {
    return;
  }
  
  // Calculate x output index with proper increment
  let x_out_idx = u32(max(0, i32(row) * incx));
  
  // Bounds checking for output array
  if (x_out_idx >= arrayLength(&x_out)) {
    return;
  }
  
  // Compute triangular matrix-vector product
  var sum = 0.0;
  
  if (trans == 0u) { // No transpose: x := A * x
    if (uplo == 1u) { // Upper triangular
      for (var col = 0u; col < n; col++) {
        let x_in_idx = u32(max(0, i32(col) * incx));
        
        if (x_in_idx < arrayLength(&x_in) && row <= col) {
          var a_val: f32;
          
          if (row == col && diag == 1u) {
            // Unit diagonal
            a_val = 1.0;
          } else if (row == col && diag == 0u) {
            // Non-unit diagonal
            let a_idx = col * lda + row;
            if (a_idx < arrayLength(&A)) {
              a_val = A[a_idx];
            } else {
              continue;
            }
          } else if (row < col) {
            // Upper triangular part
            let a_idx = col * lda + row;
            if (a_idx < arrayLength(&A)) {
              a_val = A[a_idx];
            } else {
              continue;
            }
          } else {
            // Lower part (zero)
            continue;
          }
          
          sum += a_val * x_in[x_in_idx];
        }
      }
    } else { // Lower triangular
      for (var col = 0u; col < n; col++) {
        let x_in_idx = u32(max(0, i32(col) * incx));
        
        if (x_in_idx < arrayLength(&x_in) && row >= col) {
          var a_val: f32;
          
          if (row == col && diag == 1u) {
            // Unit diagonal
            a_val = 1.0;
          } else if (row == col && diag == 0u) {
            // Non-unit diagonal
            let a_idx = col * lda + row;
            if (a_idx < arrayLength(&A)) {
              a_val = A[a_idx];
            } else {
              continue;
            }
          } else if (row > col) {
            // Lower triangular part
            let a_idx = col * lda + row;
            if (a_idx < arrayLength(&A)) {
              a_val = A[a_idx];
            } else {
              continue;
            }
          } else {
            // Upper part (zero)
            continue;
          }
          
          sum += a_val * x_in[x_in_idx];
        }
      }
    }
  } else { // Transpose: x := A^T * x
    if (uplo == 1u) { // Upper triangular (becomes lower after transpose)
      for (var col = 0u; col < n; col++) {
        let x_in_idx = u32(max(0, i32(col) * incx));
        
        if (x_in_idx < arrayLength(&x_in) && col <= row) {
          var a_val: f32;
          
          if (row == col && diag == 1u) {
            // Unit diagonal
            a_val = 1.0;
          } else if (row == col && diag == 0u) {
            // Non-unit diagonal
            let a_idx = row * lda + col;
            if (a_idx < arrayLength(&A)) {
              a_val = A[a_idx];
            } else {
              continue;
            }
          } else if (col < row) {
            // Use A^T[row,col] = A[col,row]
            let a_idx = row * lda + col;
            if (a_idx < arrayLength(&A)) {
              a_val = A[a_idx];
            } else {
              continue;
            }
          } else {
            // Upper part of transpose (zero)
            continue;
          }
          
          sum += a_val * x_in[x_in_idx];
        }
      }
    } else { // Lower triangular (becomes upper after transpose)
      for (var col = 0u; col < n; col++) {
        let x_in_idx = u32(max(0, i32(col) * incx));
        
        if (x_in_idx < arrayLength(&x_in) && col >= row) {
          var a_val: f32;
          
          if (row == col && diag == 1u) {
            // Unit diagonal
            a_val = 1.0;
          } else if (row == col && diag == 0u) {
            // Non-unit diagonal
            let a_idx = row * lda + col;
            if (a_idx < arrayLength(&A)) {
              a_val = A[a_idx];
            } else {
              continue;
            }
          } else if (col > row) {
            // Use A^T[row,col] = A[col,row]
            let a_idx = row * lda + col;
            if (a_idx < arrayLength(&A)) {
              a_val = A[a_idx];
            } else {
              continue;
            }
          } else {
            // Lower part of transpose (zero)
            continue;
          }
          
          sum += a_val * x_in[x_in_idx];
        }
      }
    }
  }
  
  // Store result
  x_out[x_out_idx] = sum;
}
