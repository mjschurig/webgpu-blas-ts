/**
 * Playwright E2E tests for SNRM2 function
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

test.describe('SNRM2 - Single Precision Euclidean Norm', () => {
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

  test('should compute norm correctly for basic vectors', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([3, 4]); // 3-4-5 triangle
      return await window.WebGPUBLAS.snrm2({ n: 2, sx });
    });
    
    // Expected: sqrt(3^2 + 4^2) = sqrt(9 + 16) = sqrt(25) = 5
    expect(result).toBeCloseTo(5.0, 4);
  });

  test('should handle unit vectors', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([1, 0, 0, 0]);
      return await window.WebGPUBLAS.snrm2({ n: 4, sx });
    });
    
    expect(result).toBeCloseTo(1.0, 4);
  });

  test('should handle zero vector', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([0, 0, 0, 0]);
      return await window.WebGPUBLAS.snrm2({ n: 4, sx });
    });
    
    expect(result).toBeCloseTo(0.0, 4);
  });

  test('should handle negative values correctly', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([-3, -4]); 
      return await window.WebGPUBLAS.snrm2({ n: 2, sx });
    });
    
    // Expected: sqrt((-3)^2 + (-4)^2) = sqrt(9 + 16) = 5
    expect(result).toBeCloseTo(5.0, 4);
  });

  test('should handle mixed positive and negative values', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([3, -4, 0, 12]); 
      return await window.WebGPUBLAS.snrm2({ n: 4, sx });
    });
    
    // Expected: sqrt(3^2 + (-4)^2 + 0^2 + 12^2) = sqrt(9 + 16 + 0 + 144) = sqrt(169) = 13
    expect(result).toBeCloseTo(13.0, 4);
  });

  test('should handle stride correctly', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([3, 999, 4, 888]); 
      return await window.WebGPUBLAS.snrm2({ n: 2, sx, incx: 2 });
    });
    
    // Expected: sqrt(3^2 + 4^2) = 5 (skipping elements at indices 1 and 3)
    expect(result).toBeCloseTo(5.0, 4);
  });

  test('should handle small values without underflow', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([1e-10, 2e-10, 3e-10]); 
      return await window.WebGPUBLAS.snrm2({ n: 3, sx });
    });
    
    // Expected: sqrt((1e-10)^2 + (2e-10)^2 + (3e-10)^2) = sqrt(14) * 1e-10
    expect(result).toBeCloseTo(Math.sqrt(14) * 1e-10, 15);
  });

  test('should handle large values without overflow', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([1e10, 2e10]); 
      return await window.WebGPUBLAS.snrm2({ n: 2, sx });
    });
    
    // Expected: sqrt((1e10)^2 + (2e10)^2) = sqrt(5) * 1e10
    // Note: Large numbers have reduced precision due to floating point limitations
    // Using relative error tolerance for very large numbers
    const expected = Math.sqrt(5) * 1e10;
    const relativeError = Math.abs(result - expected) / expected;
    expect(relativeError).toBeLessThan(1e-4); // 0.01% relative tolerance
  });

  test('should handle fractional values', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([0.6, 0.8]); 
      return await window.WebGPUBLAS.snrm2({ n: 2, sx });
    });
    
    // Expected: sqrt(0.6^2 + 0.8^2) = sqrt(0.36 + 0.64) = sqrt(1.0) = 1.0
    expect(result).toBeCloseTo(1.0, 4);
  });

  test('should handle large vectors efficiently', async ({ page }, testInfo) => {
    const { result, duration, passed } = await page.evaluate(async () => {
      const n = 10000;
      const sx = new Float32Array(n);
      
      for (let i = 0; i < n; i++) {
        sx[i] = 1.0; // All ones, norm should be sqrt(n)
      }
      
      const start = performance.now();
      const result = await window.WebGPUBLAS.snrm2({ n, sx });
      const end = performance.now();
      
      const expected = Math.sqrt(n);
      const tolerance = Math.abs(expected * 0.0001); // 0.01% tolerance
      
      return {
        result,
        duration: end - start,
        passed: Math.abs(result - expected) <= tolerance,
        expected
      };
    });
    
    expect(passed).toBe(true);
    expect(duration).toBeLessThan(1000); // Should complete in < 1 second
    
    testInfo.annotations.push({ 
      type: 'performance', 
      description: `10000 element norm computed in ${duration.toFixed(2)}ms` 
    });
  });

  test('should return 0 for edge cases', async ({ page }) => {
    const results = await page.evaluate(async () => {
      return {
        emptyVector: await window.WebGPUBLAS.snrm2({ n: 0, sx: new Float32Array([]) }),
        negativeN: await window.WebGPUBLAS.snrm2({ n: -1, sx: new Float32Array([1, 2, 3]) }),
        zeroIncx: await window.WebGPUBLAS.snrm2({ n: 3, sx: new Float32Array([1, 2, 3]), incx: 0 }),
      };
    });
    
    expect(results.emptyVector).toBe(0.0);
    expect(results.negativeN).toBe(0.0);
    expect(results.zeroIncx).toBe(0.0);
  });

  test('should be scale-invariant', async ({ page }) => {
    const results = await page.evaluate(async () => {
      const sx1 = new Float32Array([3, 4]);
      const sx2 = new Float32Array([6, 8]); // 2x scale
      
      const norm1 = await window.WebGPUBLAS.snrm2({ n: 2, sx: sx1 });
      const norm2 = await window.WebGPUBLAS.snrm2({ n: 2, sx: sx2 });
      
      return { norm1, norm2, ratio: norm2 / norm1 };
    });
    
    expect(results.norm1).toBeCloseTo(5.0, 4);
    expect(results.norm2).toBeCloseTo(10.0, 4);
    expect(results.ratio).toBeCloseTo(2.0, 4);
  });
});
