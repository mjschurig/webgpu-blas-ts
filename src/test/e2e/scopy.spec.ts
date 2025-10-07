/**
 * Playwright E2E tests for SCOPY function
 * Tests run in real browser environment with actual WebGPU shaders
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

test.describe('SCOPY - Single Precision Copy', () => {
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

  test('should copy vector x to y correctly', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([1, 2, 3, 4, 5]);
      const sy = new Float32Array([10, 20, 30, 40, 50]);
      
      await window.WebGPUBLAS.scopy({ n: 5, sx, sy });
      return Array.from(sy);
    });
    
    // Expected: y should be equal to x
    expect(result[0]).toBeCloseTo(1, 4);
    expect(result[1]).toBeCloseTo(2, 4);
    expect(result[2]).toBeCloseTo(3, 4);
    expect(result[3]).toBeCloseTo(4, 4);
    expect(result[4]).toBeCloseTo(5, 4);
  });

  test('should handle negative values correctly', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([-1, -2, -3]);
      const sy = new Float32Array([100, 200, 300]);
      
      await window.WebGPUBLAS.scopy({ n: 3, sx, sy });
      return Array.from(sy);
    });
    
    expect(result[0]).toBeCloseTo(-1, 4);
    expect(result[1]).toBeCloseTo(-2, 4);
    expect(result[2]).toBeCloseTo(-3, 4);
  });

  test('should handle stride correctly', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([1, 999, 2, 888, 3, 777]);
      const sy = new Float32Array([10, 999, 20, 888, 30, 777]);
      
      await window.WebGPUBLAS.scopy({ n: 3, sx, sy, incx: 2, incy: 2 });
      return Array.from(sy);
    });
    
    // Expected: copy elements at indices 0, 2, 4 from x to y
    expect(result[0]).toBeCloseTo(1, 4); // sx[0] -> sy[0]
    expect(result[1]).toBeCloseTo(999, 4); // unchanged
    expect(result[2]).toBeCloseTo(2, 4); // sx[2] -> sy[2]
    expect(result[3]).toBeCloseTo(888, 4); // unchanged
    expect(result[4]).toBeCloseTo(3, 4); // sx[4] -> sy[4]
    expect(result[5]).toBeCloseTo(777, 4); // unchanged
  });

  test('should handle zero and floating point values', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([0, 3.14159, -2.718, 1.414]);
      const sy = new Float32Array([1, 1, 1, 1]);
      
      await window.WebGPUBLAS.scopy({ n: 4, sx, sy });
      return Array.from(sy);
    });
    
    expect(result[0]).toBeCloseTo(0, 4);
    expect(result[1]).toBeCloseTo(3.14159, 4);
    expect(result[2]).toBeCloseTo(-2.718, 4);
    expect(result[3]).toBeCloseTo(1.414, 4);
  });

  test('should handle large vectors efficiently', async ({ page }, testInfo) => {
    const { duration, passed } = await page.evaluate(async () => {
      const n = 10000;
      const sx = new Float32Array(n);
      const sy = new Float32Array(n);
      
      for (let i = 0; i < n; i++) {
        sx[i] = Math.sin(i * 0.01);
        sy[i] = 999; // Initial value to verify copy
      }
      
      const start = performance.now();
      await window.WebGPUBLAS.scopy({ n, sx, sy });
      const end = performance.now();
      
      // Verify first and last few elements
      const firstMatch = sy[0] === sx[0];
      const lastMatch = sy[n-1] === sx[n-1];
      const middleMatch = sy[Math.floor(n/2)] === sx[Math.floor(n/2)];
      
      return {
        duration: end - start,
        passed: firstMatch && lastMatch && middleMatch
      };
    });
    
    expect(passed).toBe(true);
    expect(duration).toBeLessThan(500); // Should complete in < 500ms
    
    testInfo.annotations.push({ 
      type: 'performance', 
      description: `10000 elements copied in ${duration.toFixed(2)}ms` 
    });
  });

  test('should handle edge cases', async ({ page }) => {
    const results = await page.evaluate(async () => {
      // Test with n = 0
      const sx1 = new Float32Array([1, 2, 3]);
      const sy1 = new Float32Array([10, 20, 30]);
      const original1 = Array.from(sy1);
      await window.WebGPUBLAS.scopy({ n: 0, sx: sx1, sy: sy1 });
      
      // Test with incx = 0 (should be no-op)
      const sx2 = new Float32Array([1, 2, 3]);
      const sy2 = new Float32Array([10, 20, 30]);
      const original2 = Array.from(sy2);
      await window.WebGPUBLAS.scopy({ n: 3, sx: sx2, sy: sy2, incx: 0 });
      
      return {
        case1_unchanged: JSON.stringify(Array.from(sy1)) === JSON.stringify(original1),
        case2_unchanged: JSON.stringify(Array.from(sy2)) === JSON.stringify(original2),
      };
    });
    
    expect(results.case1_unchanged).toBe(true); // n=0 should not modify y
    expect(results.case2_unchanged).toBe(true); // incx=0 should not modify y
  });
});
