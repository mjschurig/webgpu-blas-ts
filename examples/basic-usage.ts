/**
 * Basic usage examples for webgpu-blas-ts
 */

import { dasum, initialize, isWebGPUSupported } from '../src/index.js';

async function main(): Promise<void> {
  console.log('WebGPU BLAS TypeScript Examples\n');

  // Check WebGPU support
  if (!isWebGPUSupported()) {
    console.warn('WebGPU is not supported in this environment. Using CPU fallback.');
  } else {
    console.log('WebGPU is supported! ✅');
    // Pre-initialize for better performance
    await initialize();
  }

  console.log('\n--- DASUM Examples ---');

  // Example 1: Basic usage
  console.log('\n1. Basic DASUM computation');
  const vector1 = new Float64Array([1, -2, 3, -4, 5]);
  const result1 = await dasum({ n: 5, dx: vector1 });
  console.log(`Vector: [${vector1.join(', ')}]`);
  console.log(`Sum of absolute values: ${result1}`); // Should be 15

  // Example 2: With stride
  console.log('\n2. DASUM with stride');
  const vector2 = new Float64Array([1, 999, 2, 888, 3, 777]);
  const result2 = await dasum({ n: 3, dx: vector2, incx: 2 });
  console.log(`Vector: [${vector2.join(', ')}] with stride 2`);
  console.log(`Sum of absolute values: ${result2}`); // Should be 6

  // Example 3: Large vector performance test
  console.log('\n3. Performance test with large vector');
  const largeVector = new Float64Array(10000);
  for (let i = 0; i < largeVector.length; i++) {
    largeVector[i] = Math.random() * 2 - 1; // Random values between -1 and 1
  }
  
  const start = performance.now();
  const result3 = await dasum({ n: largeVector.length, dx: largeVector });
  const end = performance.now();
  
  console.log(`Large vector (${largeVector.length} elements)`);
  console.log(`Sum of absolute values: ${result3.toFixed(4)}`);
  console.log(`Computation time: ${(end - start).toFixed(2)}ms`);

  // Example 4: Edge cases
  console.log('\n4. Edge cases');
  
  // Empty vector
  const emptyResult = await dasum({ n: 0, dx: new Float64Array([]) });
  console.log(`Empty vector result: ${emptyResult}`); // Should be 0
  
  // Single element
  const singleResult = await dasum({ n: 1, dx: new Float64Array([-42.5]) });
  console.log(`Single element [-42.5] result: ${singleResult}`); // Should be 42.5
  
  // All zeros
  const zerosResult = await dasum({ n: 5, dx: new Float64Array([0, 0, 0, 0, 0]) });
  console.log(`All zeros result: ${zerosResult}`); // Should be 0

  console.log('\n✅ All examples completed successfully!');
}

// Run examples
main().catch(console.error);












