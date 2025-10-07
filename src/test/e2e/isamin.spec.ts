/**
 * Playwright E2E tests for ISAMIN function
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

test.describe('ISAMIN - Index of Minimum Absolute Value', () => {
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

  test('should find index of minimum absolute value', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([5, -2, 8, 1]);
      return await window.WebGPUBLAS.isamin({ n: 4, sx });
    });
    
    // Expected: index of min(|5|, |-2|, |8|, |1|) = min(5, 2, 8, 1) = 1 at index 4 (1-based)
    expect(result).toBe(4);
  });

  test('should handle positive values correctly', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([3, 1, 8, 2, 4]);
      return await window.WebGPUBLAS.isamin({ n: 5, sx });
    });
    
    // Expected: min absolute value 1 at index 2 (1-based)
    expect(result).toBe(2);
  });

  test('should handle negative minimum correctly', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([5, 2, -0.5, 3, 4]);
      return await window.WebGPUBLAS.isamin({ n: 5, sx });
    });
    
    // Expected: min absolute value |-0.5| = 0.5 at index 3 (1-based)
    expect(result).toBe(3);
  });

  test('should return first index for equal minimum values', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([2, -2, 5, 2, -2]);
      return await window.WebGPUBLAS.isamin({ n: 5, sx });
    });
    
    // Expected: first occurrence of min absolute value 2 at index 1 (1-based)
    expect(result).toBe(1);
  });

  test('should handle single element vector', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([42]);
      return await window.WebGPUBLAS.isamin({ n: 1, sx });
    });
    
    expect(result).toBe(1);
  });

  test('should handle zero values correctly', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([5, 0, 3, 2]);
      return await window.WebGPUBLAS.isamin({ n: 4, sx });
    });
    
    // Expected: min absolute value 0 at index 2 (1-based)
    expect(result).toBe(2);
  });

  test('should handle all zero vector', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([0, 0, 0, 0]);
      return await window.WebGPUBLAS.isamin({ n: 4, sx });
    });
    
    // Expected: first element (index 1) when all values are equal
    expect(result).toBe(1);
  });

  test('should handle stride correctly', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([5, 999, 0.1, 888, 3, 777]); 
      return await window.WebGPUBLAS.isamin({ n: 3, sx, incx: 2 });
    });
    
    // Expected: elements at indices 0, 2, 4 are [5, 0.1, 3]
    // Min absolute value is 0.1 at position 2 in strided sequence (index 2, 1-based)
    expect(result).toBe(2);
  });

  test('should handle fractional values correctly', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([0.5, 0.8, -0.1, 0.3]);
      return await window.WebGPUBLAS.isamin({ n: 4, sx });
    });
    
    // Expected: min absolute value 0.1 at index 3 (1-based)
    expect(result).toBe(3);
  });

  test('should handle large vectors efficiently', async ({ page }, testInfo) => {
    const { result, duration, passed } = await page.evaluate(async () => {
      const n = 10000;
      const sx = new Float32Array(n);
      
      for (let i = 0; i < n; i++) {
        sx[i] = Math.abs(Math.sin(i * 0.001)) + 1; // Values between 1 and 2
      }
      // Set a known minimum value
      sx[7777] = 0.001; // This should be the minimum
      
      const start = performance.now();
      const result = await window.WebGPUBLAS.isamin({ n, sx });
      const end = performance.now();
      
      return {
        result,
        duration: end - start,
        passed: result === 7778 // 1-based index
      };
    });
    
    expect(passed).toBe(true);
    expect(duration).toBeLessThan(1000); // Should complete in < 1 second
    
    testInfo.annotations.push({ 
      type: 'performance', 
      description: `10000 element ISAMIN computed in ${duration.toFixed(2)}ms` 
    });
  });

  test('should return 0 for edge cases', async ({ page }) => {
    const results = await page.evaluate(async () => {
      return {
        emptyVector: await window.WebGPUBLAS.isamin({ n: 0, sx: new Float32Array([]) }),
        negativeN: await window.WebGPUBLAS.isamin({ n: -1, sx: new Float32Array([1, 2, 3]) }),
        zeroIncx: await window.WebGPUBLAS.isamin({ n: 3, sx: new Float32Array([1, 2, 3]), incx: 0 }),
      };
    });
    
    expect(results.emptyVector).toBe(0);
    expect(results.negativeN).toBe(0);
    expect(results.zeroIncx).toBe(0);
  });

  test('should handle very small values correctly', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([1e-5, -1e-10, 2e-8]);
      return await window.WebGPUBLAS.isamin({ n: 3, sx });
    });
    
    // Expected: min absolute value 1e-10 at index 2 (1-based)
    expect(result).toBe(2);
  });

  test('should handle very large values correctly', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([5e10, -1e10, 2e10]);
      return await window.WebGPUBLAS.isamin({ n: 3, sx });
    });
    
    // Expected: min absolute value 1e10 at index 2 (1-based)
    expect(result).toBe(2);
  });

  test('should handle mixed positive and negative values', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([-10, 5, 0.1, -0.05, 3]);
      return await window.WebGPUBLAS.isamin({ n: 5, sx });
    });
    
    // Expected: min absolute value |-0.05| = 0.05 at index 4 (1-based)
    expect(result).toBe(4);
  });

  test('should handle different stride patterns', async ({ page }) => {
    const results = await page.evaluate(async () => {
      // Test with stride 3
      const sx1 = new Float32Array([5, 99, 88, 0.1, 77, 66, 2, 55, 44]);
      const result1 = await window.WebGPUBLAS.isamin({ n: 3, sx: sx1, incx: 3 });
      
      // Test normal case for comparison
      const sx2 = new Float32Array([8, 2, 0.5, 4]);
      const result2 = await window.WebGPUBLAS.isamin({ n: 4, sx: sx2, incx: 1 });
      
      return { stride3: result1, normal: result2 };
    });
    
    // Elements at stride 3: [5, 0.1, 2], min |0.1| = 0.1 at position 2
    expect(results.stride3).toBe(2);
    // Normal case: min |0.5| = 0.5 at position 3
    expect(results.normal).toBe(3);
  });

  test('should distinguish between ISAMAX and ISAMIN', async ({ page }) => {
    const results = await page.evaluate(async () => {
      const sx = new Float32Array([0.1, 10, -0.05, 5]);
      const maxIndex = await window.WebGPUBLAS.isamax({ n: 4, sx });
      const minIndex = await window.WebGPUBLAS.isamin({ n: 4, sx });
      
      return { maxIndex, minIndex };
    });
    
    // Max: |10| = 10 at index 2, Min: |-0.05| = 0.05 at index 3
    expect(results.maxIndex).toBe(2);
    expect(results.minIndex).toBe(3);
  });
});
