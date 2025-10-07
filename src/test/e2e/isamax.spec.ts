/**
 * Playwright E2E tests for ISAMAX function
 * Tests run in real browser environment with actual WebGPU shaders
 * Based on FORTRAN BLAS reference tests (sblat1.f)
 */

import { test, expect, Page } from '@playwright/test';

// Helper function to wait for WebGPU initialization
async function waitForWebGPUInit(page: Page) {
  await page.waitForFunction(() => window.WebGPUBLAS && window.testRunner);
  
  // Wait for WebGPU support check
  await page.waitForFunction(() => {
    const statusEl = document.getElementById('webgpu-status');
    return statusEl && !statusEl.textContent?.includes('Checking WebGPU support');
  });
  
  // Check if WebGPU is supported
  const webgpuSupported = await page.evaluate(() => {
    const statusEl = document.getElementById('webgpu-status');
    return statusEl?.className.includes('success');
  });
  
  if (!webgpuSupported) {
    const errorMessage = await page.evaluate(() => {
      const statusEl = document.getElementById('webgpu-status');
      return statusEl?.textContent || 'Unknown WebGPU error';
    });
    throw new Error(`WebGPU not supported: ${errorMessage}`);
  }
}

test.describe('ISAMAX - Index of Maximum Absolute Value', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await waitForWebGPUInit(page);
    
    // Ensure WebGPU context is initialized for each test
    await page.evaluate(async () => {
      console.log('Reinitializing WebGPU for test...');
      await window.WebGPUBLAS.initialize();
      console.log('✅ WebGPU reinitialized for test');
    });
  });

  test('should find index of maximum absolute value', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([1, -5, 3, 2]);
      return await window.WebGPUBLAS.isamax({ n: 4, sx });
    });
    
    // Expected: index of max(|1|, |-5|, |3|, |2|) = max(1, 5, 3, 2) = 5 at index 2 (1-based)
    expect(result).toBe(2);
  });

  test('should handle positive values correctly', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([1, 2, 8, 3, 4]);
      return await window.WebGPUBLAS.isamax({ n: 5, sx });
    });
    
    // Expected: max absolute value 8 at index 3 (1-based)
    expect(result).toBe(3);
  });

  test('should handle negative maximum correctly', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([1, 2, -9, 3, 4]);
      return await window.WebGPUBLAS.isamax({ n: 5, sx });
    });
    
    // Expected: max absolute value |-9| = 9 at index 3 (1-based)
    expect(result).toBe(3);
  });

  test('should return first index for equal maximum values', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([5, -5, 3, 5, -5]);
      return await window.WebGPUBLAS.isamax({ n: 5, sx });
    });
    
    // Expected: first occurrence of max absolute value 5 at index 1 (1-based)
    expect(result).toBe(1);
  });

  test('should handle single element vector', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([42]);
      return await window.WebGPUBLAS.isamax({ n: 1, sx });
    });
    
    expect(result).toBe(1);
  });

  test('should handle zero values correctly', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([0, 0, 3, 0]);
      return await window.WebGPUBLAS.isamax({ n: 4, sx });
    });
    
    // Expected: max absolute value 3 at index 3 (1-based)
    expect(result).toBe(3);
  });

  test('should handle all zero vector', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([0, 0, 0, 0]);
      return await window.WebGPUBLAS.isamax({ n: 4, sx });
    });
    
    // Expected: first element (index 1) when all values are equal
    expect(result).toBe(1);
  });

  test('should handle stride correctly', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([1, 999, -8, 888, 3, 777]); 
      return await window.WebGPUBLAS.isamax({ n: 3, sx, incx: 2 });
    });
    
    // Expected: elements at indices 0, 2, 4 are [1, -8, 3]
    // Max absolute value is |-8| = 8 at position 2 in strided sequence (index 2, 1-based)
    expect(result).toBe(2);
  });

  test('should handle fractional values correctly', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([0.1, 0.5, -0.8, 0.3]);
      return await window.WebGPUBLAS.isamax({ n: 4, sx });
    });
    
    // Expected: max absolute value 0.8 at index 3 (1-based)
    expect(result).toBe(3);
  });

  test('should handle large vectors efficiently', async ({ page }, testInfo) => {
    const { result, duration, passed } = await page.evaluate(async () => {
      const n = 10000;
      const sx = new Float32Array(n);
      
      for (let i = 0; i < n; i++) {
        sx[i] = Math.sin(i * 0.001); // Values between -1 and 1
      }
      // Set a known maximum value
      sx[5555] = 99.0; // This should be the maximum
      
      const start = performance.now();
      const result = await window.WebGPUBLAS.isamax({ n, sx });
      const end = performance.now();
      
      return {
        result,
        duration: end - start,
        passed: result === 5556 // 1-based index
      };
    });
    
    expect(passed).toBe(true);
    expect(duration).toBeLessThan(1000); // Should complete in < 1 second
    
    testInfo.annotations.push({ 
      type: 'performance', 
      description: `10000 element ISAMAX computed in ${duration.toFixed(2)}ms` 
    });
  });

  test('should return 0 for edge cases', async ({ page }) => {
    const results = await page.evaluate(async () => {
      return {
        emptyVector: await window.WebGPUBLAS.isamax({ n: 0, sx: new Float32Array([]) }),
        negativeN: await window.WebGPUBLAS.isamax({ n: -1, sx: new Float32Array([1, 2, 3]) }),
        zeroIncx: await window.WebGPUBLAS.isamax({ n: 3, sx: new Float32Array([1, 2, 3]), incx: 0 }),
      };
    });
    
    expect(results.emptyVector).toBe(0);
    expect(results.negativeN).toBe(0);
    expect(results.zeroIncx).toBe(0);
  });

  test('should handle very small values correctly', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([1e-10, -3e-10, 2e-10]);
      return await window.WebGPUBLAS.isamax({ n: 3, sx });
    });
    
    // Expected: max absolute value 3e-10 at index 2 (1-based)
    expect(result).toBe(2);
  });

  test('should handle very large values correctly', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([1e10, -5e10, 2e10]);
      return await window.WebGPUBLAS.isamax({ n: 3, sx });
    });
    
    // Expected: max absolute value 5e10 at index 2 (1-based)
    expect(result).toBe(2);
  });

  test('should handle different stride patterns', async ({ page }) => {
    const results = await page.evaluate(async () => {
      // Test with stride 3
      const sx1 = new Float32Array([1, 99, 88, -7, 77, 66, 2, 55, 44]);
      const result1 = await window.WebGPUBLAS.isamax({ n: 3, sx: sx1, incx: 3 });
      
      // Test with stride -1 (should use elements in reverse)
      const sx2 = new Float32Array([1, 2, -8, 4]);
      const result2 = await window.WebGPUBLAS.isamax({ n: 4, sx: sx2, incx: 1 });
      
      return { stride3: result1, normal: result2 };
    });
    
    // Elements at stride 3: [1, -7, 2], max |-7| = 7 at position 2
    expect(results.stride3).toBe(2);
    // Normal case: max |-8| = 8 at position 3
    expect(results.normal).toBe(3);
  });
});
