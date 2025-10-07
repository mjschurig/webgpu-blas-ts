/**
 * SROTMG - Generate modified plane rotation (single precision)
 * 
 * Generates parameters for a modified Givens rotation.
 * This is a complex algorithm that's rarely used in practice.
 * 
 * Implementing as CPU-based function for efficiency.
 */

/**
 * SROTMG parameters and results
 */
export interface SROTMGParams {
  /** Input d1 (will be modified) */
  d1: Float32Array; // Single element array
  /** Input d2 (will be modified) */
  d2: Float32Array; // Single element array  
  /** Input x1 (will be modified) */
  x1: Float32Array; // Single element array
  /** Input y1 */
  y1: number;
  /** Output parameter array [flag, h11, h21, h12, h22] */
  param: Float32Array; // Must have at least 5 elements
}

/**
 * Generate modified plane rotation using CPU computation (single precision)
 */
export function srotmg(params: SROTMGParams): void {
  const { d1, d2, x1, y1, param } = params;
  
  // Validate inputs
  if (d1.length === 0 || d2.length === 0 || x1.length === 0) {
    throw new Error('SROTMG: All scalar arrays must have at least one element');
  }
  if (param.length < 5) {
    throw new Error('SROTMG: param array must have at least 5 elements');
  }
  
  let d1Val = d1[0]!;
  let d2Val = d2[0]!;
  let x1Val = x1[0]!;
  const y1Val = y1;
  
  const gam = 4096.0;
  const gamsq = gam * gam;
  const rgamsq = 1.0 / gamsq;
  
  let flag = 0.0;
  let h11 = 0.0, h12 = 0.0, h21 = 0.0, h22 = 0.0;
  
  // Handle special cases
  if (d1Val < 0.0) {
    // Set identity and return
    flag = -1.0;
    h11 = 0.0;
    h21 = 0.0;
    h12 = 0.0;
    h22 = 0.0;
    d1Val = 0.0;
    d2Val = 0.0;
    x1Val = 0.0;
  } else if (d2Val === 0.0 || y1Val === 0.0) {
    flag = -2.0; // Identity transformation
    h11 = 0.0;
    h21 = 0.0;
    h12 = 0.0;
    h22 = 0.0;
  } else {
    // General case - simplified stable algorithm
    let p2 = d2Val * y1Val;
    let p1 = d1Val * x1Val;
    
    if (Math.abs(p1) > Math.abs(p2)) {
      flag = 0.0;
      h11 = 1.0;
      h22 = 1.0;
      h21 = -y1Val / x1Val;
      h12 = p2 / p1;
      
      let u = 1.0 - h12 * h21;
      if (u > 0.0) {
        d1Val = d1Val / u;
        d2Val = d2Val / u;
        x1Val = x1Val * u;
      }
    } else {
      flag = 1.0;
      h11 = p1 / p2;
      h21 = -1.0;
      h12 = 1.0;
      h22 = x1Val / y1Val;
      
      let u = 1.0 + h11 * h22;
      let temp = d2Val / u;
      d2Val = d1Val / u;
      d1Val = temp;
      x1Val = y1Val * u;
    }
    
    // Scale to prevent overflow/underflow
    while ((d1Val !== 0.0 && Math.abs(d1Val) <= rgamsq) || Math.abs(d1Val) >= gamsq) {
      if (flag === 0.0) {
        h11 = 1.0;
        h22 = 1.0;
        flag = -1.0;
      } else {
        h21 = -1.0;
        h12 = 1.0;
        flag = -1.0;
      }
      
      if (d1Val !== 0.0) {
        if (Math.abs(d1Val) <= rgamsq) {
          d1Val = d1Val * gamsq;
          x1Val = x1Val / gam;
          h11 = h11 / gam;
          h12 = h12 / gam;
        } else {
          d1Val = d1Val / gamsq;
          x1Val = x1Val * gam;
          h11 = h11 * gam;
          h12 = h12 * gam;
        }
      }
    }
  }
  
  // Store results
  param[0] = flag;
  param[1] = h11;
  param[2] = h21;
  param[3] = h12;
  param[4] = h22;
  
  d1[0] = d1Val;
  d2[0] = d2Val;
  x1[0] = x1Val;
}

/**
 * Async wrapper for consistency with other BLAS functions
 */
export async function srotmgAsync(params: SROTMGParams): Promise<void> {
  srotmg(params);
}
