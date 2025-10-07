/**
 * Playwright E2E tests for SSCAL function
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

test.describe('SSCAL - Single Precision Scale', () => {
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

  test('should scale vector correctly with positive alpha', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([1, 2, 3, 4, 5]);
      const alpha = 3.0;
      
      await window.WebGPUBLAS.sscal({ n: 5, alpha, sx });
      return Array.from(sx);
    });
    
    // Expected: x = 3 * x = [3, 6, 9, 12, 15]
    expect(result[0]).toBeCloseTo(3, 4);
    expect(result[1]).toBeCloseTo(6, 4);
    expect(result[2]).toBeCloseTo(9, 4);
    expect(result[3]).toBeCloseTo(12, 4);
    expect(result[4]).toBeCloseTo(15, 4);
  });

  test('should handle negative alpha correctly', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([1, 2, 3, 4]);
      const alpha = -2.0;
      
      await window.WebGPUBLAS.sscal({ n: 4, alpha, sx });
      return Array.from(sx);
    });
    
    // Expected: x = -2 * x = [-2, -4, -6, -8]
    expect(result[0]).toBeCloseTo(-2, 4);
    expect(result[1]).toBeCloseTo(-4, 4);
    expect(result[2]).toBeCloseTo(-6, 4);
    expect(result[3]).toBeCloseTo(-8, 4);
  });

  test('should handle alpha = 0 (zero out vector)', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([10, -5, 3.14, -2.7]);
      const alpha = 0.0;
      
      await window.WebGPUBLAS.sscal({ n: 4, alpha, sx });
      return Array.from(sx);
    });
    
    // Expected: x = 0 * x = [0, 0, 0, 0]
    expect(result[0]).toBeCloseTo(0, 4);
    expect(result[1]).toBeCloseTo(0, 4);
    expect(result[2]).toBeCloseTo(0, 4);
    expect(result[3]).toBeCloseTo(0, 4);
  });

  test('should handle alpha = 1 (no-op)', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([1, 2, 3, 4, 5]);
      const original = Array.from(sx);
      const alpha = 1.0;
      
      await window.WebGPUBLAS.sscal({ n: 5, alpha, sx });
      return { original, result: Array.from(sx) };
    });
    
    // Expected: x should remain unchanged when alpha = 1
    expect(result.result).toEqual(result.original);
  });

  test('should handle fractional alpha correctly', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([2, 4, 6, 8]);
      const alpha = 0.5;
      
      await window.WebGPUBLAS.sscal({ n: 4, alpha, sx });
      return Array.from(sx);
    });
    
    // Expected: x = 0.5 * x = [1, 2, 3, 4]
    expect(result[0]).toBeCloseTo(1, 4);
    expect(result[1]).toBeCloseTo(2, 4);
    expect(result[2]).toBeCloseTo(3, 4);
    expect(result[3]).toBeCloseTo(4, 4);
  });

  test('should handle stride correctly', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([1, 999, 2, 888, 3, 777, 4]);
      const alpha = 10.0;
      
      await window.WebGPUBLAS.sscal({ n: 4, alpha, sx, incx: 2 });
      return Array.from(sx);
    });
    
    // Expected: scale elements at indices 0, 2, 4, 6
    expect(result[0]).toBeCloseTo(10, 4); // 1 * 10
    expect(result[1]).toBeCloseTo(999, 4); // unchanged
    expect(result[2]).toBeCloseTo(20, 4); // 2 * 10
    expect(result[3]).toBeCloseTo(888, 4); // unchanged
    expect(result[4]).toBeCloseTo(30, 4); // 3 * 10
    expect(result[5]).toBeCloseTo(777, 4); // unchanged
    expect(result[6]).toBeCloseTo(40, 4); // 4 * 10
  });

  test('should handle negative and zero values in vector', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([-3, 0, 2.5, -1.5]);
      const alpha = 4.0;
      
      await window.WebGPUBLAS.sscal({ n: 4, alpha, sx });
      return Array.from(sx);
    });
    
    // Expected: x = 4 * x = [-12, 0, 10, -6]
    expect(result[0]).toBeCloseTo(-12, 4);
    expect(result[1]).toBeCloseTo(0, 4);
    expect(result[2]).toBeCloseTo(10, 4);
    expect(result[3]).toBeCloseTo(-6, 4);
  });

  test('should handle large vectors efficiently', async ({ page }, testInfo) => {
    const { duration, passed } = await page.evaluate(async () => {
      const n = 10000;
      const sx = new Float32Array(n);
      const alpha = 2.5;
      
      // Initialize with known values
      for (let i = 0; i < n; i++) {
        sx[i] = i + 1;
      }
      
      const start = performance.now();
      await window.WebGPUBLAS.sscal({ n, alpha, sx });
      const end = performance.now();
      
      // Verify first and last few elements
      const firstCorrect = Math.abs(sx[0] - 2.5) < 1e-4; // 1 * 2.5
      const middleCorrect = Math.abs(sx[4999] - 12500) < 1e-4; // 5000 * 2.5
      const lastCorrect = Math.abs(sx[9999] - 25000) < 1e-4; // 10000 * 2.5
      
      return {
        duration: end - start,
        passed: firstCorrect && middleCorrect && lastCorrect
      };
    });
    
    expect(passed).toBe(true);
    expect(duration).toBeLessThan(500); // Should complete in < 500ms
    
    testInfo.annotations.push({ 
      type: 'performance', 
      description: `10000 elements scaled in ${duration.toFixed(2)}ms` 
    });
  });

  test('should handle edge cases', async ({ page }) => {
    const results = await page.evaluate(async () => {
      // Test with n = 0
      const sx1 = new Float32Array([1, 2, 3]);
      const original1 = Array.from(sx1);
      await window.WebGPUBLAS.sscal({ n: 0, alpha: 5.0, sx: sx1 });
      
      // Test with incx = 0 (should be no-op)
      const sx2 = new Float32Array([1, 2, 3]);
      const original2 = Array.from(sx2);
      await window.WebGPUBLAS.sscal({ n: 3, alpha: 5.0, sx: sx2, incx: 0 });
      
      return {
        case1_unchanged: JSON.stringify(Array.from(sx1)) === JSON.stringify(original1),
        case2_unchanged: JSON.stringify(Array.from(sx2)) === JSON.stringify(original2),
      };
    });
    
    expect(results.case1_unchanged).toBe(true); // n=0 should not modify x
    expect(results.case2_unchanged).toBe(true); // incx=0 should not modify x
  });
});
