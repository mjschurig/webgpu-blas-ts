/**
 * SROTM - Apply modified plane rotation (single precision)
 * 
 * Applies a modified Givens rotation with flag-based parameter selection.
 * This is a more complex rotation that includes scaling factors.
 * 
 * Since this is rarely used and complex, implementing as CPU-based function.
 */

/**
 * SROTM parameters
 */
export interface SROTMParams {
  /** Number of elements in the vectors */
  n: number;
  /** First vector x (modified in-place) */
  sx: Float32Array;
  /** Increment/stride between elements in x (default: 1) */
  incx?: number;
  /** Second vector y (modified in-place) */
  sy: Float32Array;
  /** Increment/stride between elements in y (default: 1) */
  incy?: number;
  /** Parameter array [flag, h11, h21, h12, h22] */
  param: Float32Array;
}

/**
 * Apply modified plane rotation using CPU computation (single precision)
 */
export function srotm(params: SROTMParams): void {
  const { n, sx, sy, incx = 1, incy = 1, param } = params;
  
  // Validate inputs
  if (n <= 0) return;
  if (incx === 0 || incy === 0) return;
  if (sx.length === 0 || sy.length === 0) return;
  if (param.length < 5) {
    throw new Error('SROTM: param array must have at least 5 elements');
  }
  
  const flag = param[0];
  
  // Early return for flag = -2 (identity transformation)
  if (flag === -2.0) return;
  
  let h11 = 0.0, h12 = 0.0, h21 = 0.0, h22 = 0.0;
  
  // Set parameters based on flag
  if (flag === -1.0) {
    // Use all parameters from param array
    h11 = param[1]!;
    h21 = param[2]!;
    h12 = param[3]!;
    h22 = param[4]!;
  } else if (flag === 0.0) {
    // h11 = 1, h22 = 1, use h21 and h12
    h11 = 1.0;
    h21 = param[2]!;
    h12 = param[3]!;
    h22 = 1.0;
  } else if (flag === 1.0) {
    // h21 = -1, h12 = 1, use h11 and h22
    h11 = param[1]!;
    h21 = -1.0;
    h12 = 1.0;
    h22 = param[4]!;
  } else {
    // Invalid flag
    return;
  }
  
  // Apply transformation to vectors
  let ix = incx > 0 ? 0 : (n - 1) * (-incx);
  let iy = incy > 0 ? 0 : (n - 1) * (-incy);
  
  for (let i = 0; i < n; i++) {
    if (ix >= 0 && ix < sx.length && iy >= 0 && iy < sy.length) {
      const tempx = h11 * sx[ix]! + h12 * sy[iy]!;
      const tempy = h21 * sx[ix]! + h22 * sy[iy]!;
      
      sx[ix] = tempx;
      sy[iy] = tempy;
    }
    
    ix += incx;
    iy += incy;
  }
}

/**
 * Async wrapper for consistency with other BLAS functions
 */
export async function srotmAsync(params: SROTMParams): Promise<void> {
  srotm(params);
}
