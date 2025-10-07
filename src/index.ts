/**
 * WebGPU BLAS TypeScript Library
 * 
 * A WebGPU-accelerated implementation of Basic Linear Algebra Subprograms (BLAS)
 */

// WebGPU Context
export { WebGPUContext, getWebGPUContext } from './webgpu/context';

// BLAS Level 1 Functions (Single Precision Only - WebGPU limitation)
export { sasum, type SASUMParams } from './blas/level1/sasum';
export { saxpy, type SAXPYParams } from './blas/level1/saxpy';
export { scopy, type SCOPYParams } from './blas/level1/scopy';
export { sscal, type SSCALParams } from './blas/level1/sscal';
export { sswap, type SSWAPParams } from './blas/level1/sswap';
export { sdot, type SDOTParams } from './blas/level1/sdot';
export { snrm2, type SNRM2Params } from './blas/level1/snrm2';
export { isamax, type ISAMAXParams } from './blas/level1/isamax';
export { isamin, type ISAMINParams } from './blas/level1/isamin';
export { srot, type SROTParams } from './blas/level1/srot';
export { srotg, srotgAsync, type SROTGParams } from './blas/level1/srotg';
export { srotm, srotmAsync, type SROTMParams } from './blas/level1/srotm';
export { srotmg, srotmgAsync, type SROTMGParams } from './blas/level1/srotmg';

// BLAS Level 2 Functions (Single Precision Only - WebGPU limitation)
export { sgemv, type SGEMVParams, type GEMVTrans } from './blas/level2/sgemv';
export { sger, type SGERParams } from './blas/level2/sger';
export { ssymv, type SSYMVParams, type SYMVUplo } from './blas/level2/ssymv';
export { ssyr, type SSYRParams } from './blas/level2/ssyr';
export { strmv, type STRMVParams, type TRMVUplo, type TRMVTrans, type TRMVDiag } from './blas/level2/strmv';
export { strsv, type STRSVParams } from './blas/level2/strsv';
export { ssyr2, type SSYR2Params } from './blas/level2/ssyr2';
export { sgbmv, type SGBMVParams } from './blas/level2/sgbmv';

// Library information
export const VERSION = '0.1.0';
export const DESCRIPTION = 'WebGPU-accelerated BLAS implementation in TypeScript';

/**
 * Check if WebGPU is supported in the current environment
 */
export function isWebGPUSupported(): boolean {
  return typeof navigator !== 'undefined' && 'gpu' in navigator;
}

/**
 * Initialize the library (optional, context is lazily initialized)
 */
export async function initialize(): Promise<void> {
  if (!isWebGPUSupported()) {
    throw new Error('WebGPU is not supported in this environment');
  }
  
  // Import and pre-initialize the context
  const { getWebGPUContext } = await import('./webgpu/context');
  try {
    await getWebGPUContext();
    console.log('✅ WebGPU context initialized successfully');
  } catch (error) {
    console.error('❌ WebGPU initialization failed:', error);
    throw error;
  }
}
















