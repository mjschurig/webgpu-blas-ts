/**
 * SROTG - Generate plane rotation (single precision)
 * 
 * Generates a Givens plane rotation that zeros out the second component:
 * Given scalars a and b, computes c (cosine), s (sine), r (hypotenuse), and z (reconstruction info)
 * such that: [c s; -s c] * [a; b] = [r; 0]
 * 
 * This is a WebGPU BLAS implementation of the reference SROTG function.
 * Since this is a scalar operation, it's implemented on the CPU for efficiency.
 */

/**
 * SROTG parameters and results
 */
export interface SROTGParams {
  /** Input value a (will be modified to contain r) */
  a: Float32Array; // Single element array for in-place modification
  /** Input value b (will be modified to contain z) */
  b: Float32Array; // Single element array for in-place modification
  /** Output cosine value */
  c: Float32Array; // Single element array to store result
  /** Output sine value */
  s: Float32Array; // Single element array to store result
}

/**
 * Generate plane rotation using CPU computation (single precision)
 * Modifies a, b, c, and s arrays in-place.
 */
export function srotg(params: SROTGParams): void {
  const { a, b, c, s } = params;
  
  // Validate inputs
  if (a.length === 0 || b.length === 0 || c.length === 0 || s.length === 0) {
    throw new Error('SROTG: All parameter arrays must have at least one element');
  }
  
  const aVal = a[0]!;
  const bVal = b[0]!;
  
  let cVal = 0.0;
  let sVal = 0.0;
  let rVal = 0.0;
  let zVal = 0.0;
  
  // Handle special cases
  if (bVal === 0.0) {
    // b is zero, no rotation needed
    cVal = 1.0;
    sVal = 0.0;
    rVal = aVal;
    zVal = 0.0;
  } else if (aVal === 0.0) {
    // a is zero, 90-degree rotation
    cVal = 0.0;
    sVal = 1.0;
    rVal = bVal;
    zVal = 1.0;
  } else {
    // General case using stable algorithm
    const absA = Math.abs(aVal);
    const absB = Math.abs(bVal);
    
    if (absA > absB) {
      // |a| > |b| case
      const t = bVal / aVal;
      const u = Math.sign(aVal) * Math.sqrt(1.0 + t * t);
      cVal = 1.0 / u;
      sVal = cVal * t;
      rVal = aVal * u;
      zVal = sVal;
    } else {
      // |b| >= |a| case
      const t = aVal / bVal;
      const u = Math.sign(bVal) * Math.sqrt(1.0 + t * t);
      sVal = 1.0 / u;
      cVal = sVal * t;
      rVal = bVal * u;
      zVal = (cVal !== 0.0) ? (1.0 / cVal) : 1.0;
    }
  }
  
  // Store results back to arrays
  a[0] = rVal; // a is overwritten with r (hypotenuse)
  b[0] = zVal; // b is overwritten with z (reconstruction info)
  c[0] = cVal; // cosine
  s[0] = sVal; // sine
}

/**
 * Async wrapper for consistency with other BLAS functions
 */
export async function srotgAsync(params: SROTGParams): Promise<void> {
  srotg(params);
}
