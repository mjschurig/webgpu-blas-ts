/**
 * Example: Using WGSL shader imports with esbuild-plugin-glsl
 * 
 * This example shows how to use the new WGSL import system
 * instead of string literals for WebGPU shaders.
 */

import { dasum, initialize, isWebGPUSupported } from '../src/index';

// The dasum function now uses WGSL files imported like this:
// import dasumStage1Shader from '../shaders/dasum_stage1.wgsl';
// import dasumStage2Shader from '../shaders/dasum_stage2.wgsl';

async function exampleWithWGSLImports(): Promise<void> {
  console.log('🚀 WebGPU BLAS with WGSL Imports Example\n');

  // Check WebGPU support
  if (!isWebGPUSupported()) {
    console.warn('⚠️  WebGPU is not supported in this environment');
    return;
  }

  console.log('✅ WebGPU is supported!');
  
  // Initialize the library
  try {
    await initialize();
    console.log('✅ WebGPU context initialized');
  } catch (error) {
    console.error('❌ Failed to initialize WebGPU:', error);
    return;
  }

  // Example 1: Basic DASUM computation
  console.log('\n📊 Example 1: Basic DASUM computation');
  const vector1 = new Float64Array([1, -2, 3, -4, 5, -6, 7, -8, 9, -10]);
  console.log('Input vector:', Array.from(vector1));
  
  const result1 = await dasum({ n: 10, dx: vector1 });
  console.log(`Sum of absolute values: ${result1}`);
  console.log(`Expected: ${1+2+3+4+5+6+7+8+9+10} ✅`);

  // Example 2: Large vector performance
  console.log('\n⚡ Example 2: Large vector performance');
  const largeSize = 100000;
  const largeVector = new Float64Array(largeSize);
  
  // Fill with random values
  for (let i = 0; i < largeSize; i++) {
    largeVector[i] = Math.random() * 2 - 1; // Random between -1 and 1
  }
  
  console.log(`Processing ${largeSize} elements...`);
  const start = performance.now();
  const result2 = await dasum({ n: largeSize, dx: largeVector });
  const end = performance.now();
  
  console.log(`Result: ${result2.toFixed(4)}`);
  console.log(`Computation time: ${(end - start).toFixed(2)}ms`);
  console.log('🎯 This computation used WGSL shaders loaded from external files!');

  // Example 3: Stride functionality
  console.log('\n🔢 Example 3: Stride functionality');
  const stridedVector = new Float64Array([1, 999, 2, 888, 3, 777, 4, 666]);
  console.log('Input vector:', Array.from(stridedVector));
  
  const result3 = await dasum({ n: 4, dx: stridedVector, incx: 2 });
  console.log(`Sum with stride 2: ${result3}`);
  console.log('Elements used: 1, 2, 3, 4 (skipping 999, 888, 777, 666)');
  console.log(`Expected: ${1+2+3+4} ✅`);

  console.log('\n🎉 All examples completed successfully!');
  console.log('\n💡 Key Benefits of WGSL Imports:');
  console.log('   • Syntax highlighting in editors');
  console.log('   • Automatic shader minification');
  console.log('   • Better maintainability');
  console.log('   • Type-safe imports');
  console.log('   • Clean separation of concerns');
}

// Run the example
if (typeof window !== 'undefined') {
  // Browser environment
  exampleWithWGSLImports().catch(console.error);
} else {
  // Node.js environment
  console.log('This example is designed for browser environments with WebGPU support.');
  console.log('To run this example:');
  console.log('1. Build the project: npm run build');
  console.log('2. Serve the dist folder with a local web server');
  console.log('3. Open in a WebGPU-compatible browser (Chrome 113+)');
}

export { exampleWithWGSLImports };
