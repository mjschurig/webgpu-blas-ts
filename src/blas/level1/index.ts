/**
 * Level 1 BLAS operations
 * 
 * This module provides WebGPU implementations of Level 1 BLAS operations,
 * supporting single precision (WebGPU limitation).
 */

// ASUM - Sum of absolute values
export { sasum, type SASUMParams } from './sasum';

// AXPY - Compute y = alpha * x + y  
export { saxpy, type SAXPYParams } from './saxpy';

// COPY - Copy vector x to y
export { scopy, type SCOPYParams } from './scopy';

// SCAL - Scale vector by scalar
export { sscal, type SSCALParams } from './sscal';

// SWAP - Swap two vectors
export { sswap, type SSWAPParams } from './sswap';

// DOT - Dot product of two vectors
export { sdot, type SDOTParams } from './sdot';

// NRM2 - Euclidean norm
export { snrm2, type SNRM2Params } from './snrm2';

// IAMAX - Index of maximum absolute value
export { isamax, type ISAMAXParams } from './isamax';

// IAMIN - Index of minimum absolute value
export { isamin, type ISAMINParams } from './isamin';

// ROT - Apply plane rotation
export { srot, type SROTParams } from './srot';

// ROTG - Generate plane rotation 
export { srotg, srotgAsync, type SROTGParams } from './srotg';

// ROTM - Apply modified plane rotation
export { srotm, srotmAsync, type SROTMParams } from './srotm';

// ROTMG - Generate modified plane rotation
export { srotmg, srotmgAsync, type SROTMGParams } from './srotmg';
