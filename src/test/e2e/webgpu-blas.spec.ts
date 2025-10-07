/**
 * Playwright E2E tests for WebGPU BLAS functions
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

test.describe('WebGPU BLAS E2E Tests', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await waitForWebGPUInit(page);
  });

  test.describe('DASUM - Double Precision Absolute Sum', () => {
    test('should compute sum of absolute values for positive numbers', async ({ page }) => {
      const result = await page.evaluate(async () => {
        const dx = new Float64Array([1, 2, 3, 4, 5]);
        return await window.WebGPUBLAS.dasum({ n: 5, dx });
      });
      
      expect(result).toBeCloseTo(15.0, 5);
    });

    test('should compute sum of absolute values for negative numbers', async ({ page }) => {
      const result = await page.evaluate(async () => {
        const dx = new Float64Array([-1, -2, -3, -4, -5]);
        return await window.WebGPUBLAS.dasum({ n: 5, dx });
      });
      
      expect(result).toBeCloseTo(15.0, 5);
    });

    test('should compute sum of absolute values for mixed signs', async ({ page }) => {
      const result = await page.evaluate(async () => {
        const dx = new Float64Array([1, -2, 3, -4, 5]);
        return await window.WebGPUBLAS.dasum({ n: 5, dx });
      });
      
      expect(result).toBeCloseTo(15.0, 5);
    });

    test('should handle stride correctly', async ({ page }) => {
      const result = await page.evaluate(async () => {
        const dx = new Float64Array([1, 999, 2, 888, 3, 777]);
        return await window.WebGPUBLAS.dasum({ n: 3, dx, incx: 2 });
      });
      
      expect(result).toBeCloseTo(6.0, 5); // |1| + |2| + |3|
    });

    test('should handle large vectors efficiently', async ({ page }, testInfo) => {
      const { result, duration } = await page.evaluate(async () => {
        const n = 10000;
        const dx = new Float64Array(n);
        for (let i = 0; i < n; i++) {
          dx[i] = (i % 2 === 0) ? 1 : -1;
        }
        
        const start = performance.now();
        const result = await window.WebGPUBLAS.dasum({ n, dx });
        const end = performance.now();
        
        return { result, duration: end - start };
      });
      
      expect(result).toBeCloseTo(10000, 5);
      expect(duration).toBeLessThan(1000); // Should complete in < 1 second
      
      // Add performance info to test results
      testInfo.annotations.push({ 
        type: 'performance', 
        description: `Completed in ${duration.toFixed(2)}ms` 
      });
    });

    test('should return 0 for edge cases', async ({ page }) => {
      const results = await page.evaluate(async () => {
        return {
          emptyVector: await window.WebGPUBLAS.dasum({ n: 0, dx: new Float64Array([]) }),
          negativeN: await window.WebGPUBLAS.dasum({ n: -1, dx: new Float64Array([1, 2, 3]) }),
          zeroIncx: await window.WebGPUBLAS.dasum({ n: 3, dx: new Float64Array([1, 2, 3]), incx: 0 }),
        };
      });
      
      expect(results.emptyVector).toBe(0.0);
      expect(results.negativeN).toBe(0.0);
      expect(results.zeroIncx).toBe(0.0);
    });
  });

  test.describe('SASUM - Single Precision Absolute Sum', () => {
    test('should compute sum of absolute values for mixed signs', async ({ page }) => {
      const result = await page.evaluate(async () => {
        const sx = new Float32Array([1, -2, 3, -4, 5]);
        return await window.WebGPUBLAS.sasum({ n: 5, sx });
      });
      
      expect(result).toBeCloseTo(15.0, 4); // Less precision for float32
    });

    test('should handle stride correctly', async ({ page }) => {
      const result = await page.evaluate(async () => {
        const sx = new Float32Array([1, 999, 2, 888, 3, 777]);
        return await window.WebGPUBLAS.sasum({ n: 3, sx, incx: 2 });
      });
      
      expect(result).toBeCloseTo(6.0, 4);
    });
  });

  test.describe('DAXPY - Double Precision A*X Plus Y', () => {
    test('should compute y = alpha * x + y correctly', async ({ page }) => {
      const result = await page.evaluate(async () => {
        const dx = new Float64Array([1, 2, 3, 4]);
        const dy = new Float64Array([5, 6, 7, 8]);
        const alpha = 2.0;
        
        await window.WebGPUBLAS.daxpy({ n: 4, alpha, dx, dy });
        return Array.from(dy);
      });
      
      // Expected: y = 2*x + y = [2*1+5, 2*2+6, 2*3+7, 2*4+8] = [7, 10, 13, 16]
      expect(result[0]).toBeCloseTo(7, 5);
      expect(result[1]).toBeCloseTo(10, 5);
      expect(result[2]).toBeCloseTo(13, 5);
      expect(result[3]).toBeCloseTo(16, 5);
    });

    test('should handle alpha = 0 (no-op)', async ({ page }) => {
      const result = await page.evaluate(async () => {
        const dx = new Float64Array([1, 2, 3, 4]);
        const dy = new Float64Array([5, 6, 7, 8]);
        const original = Array.from(dy);
        
        await window.WebGPUBLAS.daxpy({ n: 4, alpha: 0.0, dx, dy });
        return { original, result: Array.from(dy) };
      });
      
      // y should remain unchanged when alpha = 0
      expect(result.result).toEqual(result.original);
    });

    test('should handle negative alpha', async ({ page }) => {
      const result = await page.evaluate(async () => {
        const dx = new Float64Array([1, 2, 3, 4]);
        const dy = new Float64Array([5, 6, 7, 8]);
        const alpha = -1.0;
        
        await window.WebGPUBLAS.daxpy({ n: 4, alpha, dx, dy });
        return Array.from(dy);
      });
      
      // Expected: y = -x + y = [-1+5, -2+6, -3+7, -4+8] = [4, 4, 4, 4]
      expect(result[0]).toBeCloseTo(4, 5);
      expect(result[1]).toBeCloseTo(4, 5);
      expect(result[2]).toBeCloseTo(4, 5);
      expect(result[3]).toBeCloseTo(4, 5);
    });

    test('should handle different increments', async ({ page }) => {
      const result = await page.evaluate(async () => {
        const dx = new Float64Array([1, -999, 2, -999, 3, -999, 4]);
        const dy = new Float64Array([10, -999, 20, -999, 30, -999, 40]);
        const alpha = 2.0;
        
        await window.WebGPUBLAS.daxpy({ n: 4, alpha, dx, dy, incx: 2, incy: 2 });
        return Array.from(dy);
      });
      
      // Expected: y = 2*x + y at indices 0, 2, 4, 6
      expect(result[0]).toBeCloseTo(12, 5); // 2*1 + 10
      expect(result[1]).toBeCloseTo(-999, 5); // unchanged
      expect(result[2]).toBeCloseTo(24, 5); // 2*2 + 20
      expect(result[3]).toBeCloseTo(-999, 5); // unchanged
      expect(result[4]).toBeCloseTo(36, 5); // 2*3 + 30
      expect(result[5]).toBeCloseTo(-999, 5); // unchanged
      expect(result[6]).toBeCloseTo(48, 5); // 2*4 + 40
    });
  });

  test.describe('SAXPY - Single Precision A*X Plus Y', () => {
    test('should compute y = alpha * x + y correctly', async ({ page }) => {
      const result = await page.evaluate(async () => {
        const sx = new Float32Array([1, 2, 3, 4]);
        const sy = new Float32Array([5, 6, 7, 8]);
        const alpha = 2.0;
        
        await window.WebGPUBLAS.saxpy({ n: 4, alpha, sx, sy });
        return Array.from(sy);
      });
      
      expect(result[0]).toBeCloseTo(7, 4);
      expect(result[1]).toBeCloseTo(10, 4);
      expect(result[2]).toBeCloseTo(13, 4);
      expect(result[3]).toBeCloseTo(16, 4);
    });
  });

  test.describe('Performance Tests', () => {
    test('should handle medium-sized vectors efficiently', async ({ page }, testInfo) => {
      const { duration, passed } = await page.evaluate(async () => {
        const n = 5000;
        const dx = new Float64Array(n);
        for (let i = 0; i < n; i++) {
          dx[i] = Math.random() * 2 - 1; // Random between -1 and 1
        }
        
        const start = performance.now();
        const result = await window.WebGPUBLAS.dasum({ n, dx });
        const end = performance.now();
        
        return {
          duration: end - start,
          passed: result > 0 && result < n * 2 // Reasonable bounds check
        };
      });
      
      expect(passed).toBe(true);
      expect(duration).toBeLessThan(500); // Should complete in < 500ms
      
      testInfo.annotations.push({ 
        type: 'performance', 
        description: `5000 elements processed in ${duration.toFixed(2)}ms` 
      });
    });
  });
});
