/**
 * WebGPU BLAS TypeScript Library
 * 
 * A WebGPU-accelerated implementation of Basic Linear Algebra Subprograms (BLAS)
 */

// WebGPU Context
export { WebGPUContext, getWebGPUContext } from './webgpu/context';

// BLAS Level 1 Functions
export { dasum, type DASUMParams } from './blas/level1/dasum';

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
  await getWebGPUContext();
}












