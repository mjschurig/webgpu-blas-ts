/**
 * Playwright E2E tests for SDOT function
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

test.describe('SDOT - Single Precision Dot Product', () => {
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

  test('should compute basic dot product correctly', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([1, 2, 3, 4]);
      const sy = new Float32Array([2, 3, 4, 5]);
      return await window.WebGPUBLAS.sdot({ n: 4, sx, sy });
    });
    
    // Expected: 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
    expect(result).toBeCloseTo(40.0, 4);
  });

  test('should handle zero vectors', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([0, 0, 0, 0]);
      const sy = new Float32Array([1, 2, 3, 4]);
      return await window.WebGPUBLAS.sdot({ n: 4, sx, sy });
    });
    
    expect(result).toBeCloseTo(0.0, 4);
  });

  test('should handle negative values', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([1, -2, 3, -4]);
      const sy = new Float32Array([2, 3, -4, 5]);
      return await window.WebGPUBLAS.sdot({ n: 4, sx, sy });
    });
    
    // Expected: 1*2 + (-2)*3 + 3*(-4) + (-4)*5 = 2 - 6 - 12 - 20 = -36
    expect(result).toBeCloseTo(-36.0, 4);
  });

  test('should handle different increments', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([1, 999, 2, 888, 3, 777, 4]);
      const sy = new Float32Array([5, 999, 6, 888, 7, 777, 8]);
      return await window.WebGPUBLAS.sdot({ n: 4, sx, sy, incx: 2, incy: 2 });
    });
    
    // Expected: 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
    expect(result).toBeCloseTo(70.0, 4);
  });

  test('should handle single element', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([3.5]);
      const sy = new Float32Array([2.0]);
      return await window.WebGPUBLAS.sdot({ n: 1, sx, sy });
    });
    
    expect(result).toBeCloseTo(7.0, 4);
  });

  test('should handle large vectors efficiently', async ({ page }, testInfo) => {
    const { result, duration, passed } = await page.evaluate(async () => {
      const n = 10000;
      const sx = new Float32Array(n);
      const sy = new Float32Array(n);
      
      for (let i = 0; i < n; i++) {
        sx[i] = (i + 1) / 1000; // 0.001, 0.002, ...
        sy[i] = (i + 1) / 2000; // 0.0005, 0.001, ...
      }
      
      const start = performance.now();
      const result = await window.WebGPUBLAS.sdot({ n, sx, sy });
      const end = performance.now();
      
      // Expected result can be computed: sum((i+1)/1000 * (i+1)/2000) for i=0 to 9999
      // This is sum((i+1)^2) * (1/2000000) = (n*(n+1)*(2*n+1)/6) * (1/2000000)
      const expected = (n * (n + 1) * (2 * n + 1) / 6) * (1 / 2000000);
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
      description: `10000 element dot product computed in ${duration.toFixed(2)}ms` 
    });
  });

  test('should handle floating point precision correctly', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([0.1, 0.2, 0.3]);
      const sy = new Float32Array([0.4, 0.5, 0.6]);
      return await window.WebGPUBLAS.sdot({ n: 3, sx, sy });
    });
    
    // Expected: 0.1*0.4 + 0.2*0.5 + 0.3*0.6 = 0.04 + 0.1 + 0.18 = 0.32
    expect(result).toBeCloseTo(0.32, 4);
  });

  test('should return 0 for edge cases', async ({ page }) => {
    const results = await page.evaluate(async () => {
      return {
        emptyVector: await window.WebGPUBLAS.sdot({ n: 0, sx: new Float32Array([]), sy: new Float32Array([]) }),
        negativeN: await window.WebGPUBLAS.sdot({ n: -1, sx: new Float32Array([1, 2, 3]), sy: new Float32Array([1, 2, 3]) }),
        zeroIncx: await window.WebGPUBLAS.sdot({ n: 3, sx: new Float32Array([1, 2, 3]), sy: new Float32Array([1, 2, 3]), incx: 0 }),
        zeroIncy: await window.WebGPUBLAS.sdot({ n: 3, sx: new Float32Array([1, 2, 3]), sy: new Float32Array([1, 2, 3]), incy: 0 }),
      };
    });
    
    expect(results.emptyVector).toBe(0.0);
    expect(results.negativeN).toBe(0.0);
    expect(results.zeroIncx).toBe(0.0);
    expect(results.zeroIncy).toBe(0.0);
  });

  test('should handle orthogonal vectors', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([1, 0, 0]);
      const sy = new Float32Array([0, 1, 0]);
      return await window.WebGPUBLAS.sdot({ n: 3, sx, sy });
    });
    
    expect(result).toBeCloseTo(0.0, 4);
  });
});
