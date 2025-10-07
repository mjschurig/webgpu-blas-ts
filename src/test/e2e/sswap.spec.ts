/**
 * Playwright E2E tests for SSWAP function
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

test.describe('SSWAP - Single Precision Swap', () => {
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

  test('should swap vectors correctly', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([1, 2, 3, 4]);
      const sy = new Float32Array([5, 6, 7, 8]);
      const originalX = Array.from(sx);
      const originalY = Array.from(sy);
      
      await window.WebGPUBLAS.sswap({ n: 4, sx, sy });
      
      return {
        newX: Array.from(sx),
        newY: Array.from(sy),
        originalX,
        originalY
      };
    });
    
    // After swap: sx should have original sy values, sy should have original sx values
    expect(result.newX).toEqual(result.originalY);
    expect(result.newY).toEqual(result.originalX);
  });

  test('should handle negative values correctly', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([-1, -2, -3]);
      const sy = new Float32Array([1, 2, 3]);
      
      await window.WebGPUBLAS.sswap({ n: 3, sx, sy });
      
      return {
        sx: Array.from(sx),
        sy: Array.from(sy)
      };
    });
    
    expect(result.sx).toEqual([1, 2, 3]);
    expect(result.sy).toEqual([-1, -2, -3]);
  });

  test('should handle zero values correctly', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([0, 0, 0]);
      const sy = new Float32Array([1, 2, 3]);
      
      await window.WebGPUBLAS.sswap({ n: 3, sx, sy });
      
      return {
        sx: Array.from(sx),
        sy: Array.from(sy)
      };
    });
    
    expect(result.sx).toEqual([1, 2, 3]);
    expect(result.sy).toEqual([0, 0, 0]);
  });

  test('should handle stride correctly', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([1, 999, 2, 888, 3, 777]);
      const sy = new Float32Array([10, 999, 20, 888, 30, 777]);
      
      await window.WebGPUBLAS.sswap({ n: 3, sx, sy, incx: 2, incy: 2 });
      
      return {
        sx: Array.from(sx),
        sy: Array.from(sy)
      };
    });
    
    // Should swap elements at indices 0, 2, 4
    expect(result.sx[0]).toBeCloseTo(10, 4);  // sx[0] gets sy[0]
    expect(result.sx[1]).toBeCloseTo(999, 4); // unchanged
    expect(result.sx[2]).toBeCloseTo(20, 4);  // sx[2] gets sy[2]  
    expect(result.sx[3]).toBeCloseTo(888, 4); // unchanged
    expect(result.sx[4]).toBeCloseTo(30, 4);  // sx[4] gets sy[4]
    expect(result.sx[5]).toBeCloseTo(777, 4); // unchanged
    
    expect(result.sy[0]).toBeCloseTo(1, 4);   // sy[0] gets sx[0]
    expect(result.sy[1]).toBeCloseTo(999, 4); // unchanged
    expect(result.sy[2]).toBeCloseTo(2, 4);   // sy[2] gets sx[2]
    expect(result.sy[3]).toBeCloseTo(888, 4); // unchanged
    expect(result.sy[4]).toBeCloseTo(3, 4);   // sy[4] gets sx[4]
    expect(result.sy[5]).toBeCloseTo(777, 4); // unchanged
  });

  test('should handle different increments', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([1, 2, 3, 4]);
      const sy = new Float32Array([10, -999, 20, -888, 30]);
      
      await window.WebGPUBLAS.sswap({ n: 2, sx, sy, incx: 1, incy: 2 });
      
      return {
        sx: Array.from(sx),
        sy: Array.from(sy)
      };
    });
    
    expect(result.sx[0]).toBeCloseTo(10, 4); // sx[0] gets sy[0]
    expect(result.sx[1]).toBeCloseTo(20, 4); // sx[1] gets sy[2]
    expect(result.sx[2]).toBeCloseTo(3, 4);  // unchanged
    expect(result.sx[3]).toBeCloseTo(4, 4);  // unchanged
    
    expect(result.sy[0]).toBeCloseTo(1, 4);    // sy[0] gets sx[0]
    expect(result.sy[1]).toBeCloseTo(-999, 4); // unchanged
    expect(result.sy[2]).toBeCloseTo(2, 4);    // sy[2] gets sx[1]
    expect(result.sy[3]).toBeCloseTo(-888, 4); // unchanged
    expect(result.sy[4]).toBeCloseTo(30, 4);   // unchanged
  });

  test('should handle single element correctly', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([42.0]);
      const sy = new Float32Array([13.0]);
      
      await window.WebGPUBLAS.sswap({ n: 1, sx, sy });
      
      return {
        sx: Array.from(sx),
        sy: Array.from(sy)
      };
    });
    
    expect(result.sx[0]).toBeCloseTo(13.0, 4);
    expect(result.sy[0]).toBeCloseTo(42.0, 4);
  });

  test('should handle floating point values correctly', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([3.14, -2.71, 1.41]);
      const sy = new Float32Array([2.30, 6.28, -1.73]);
      
      await window.WebGPUBLAS.sswap({ n: 3, sx, sy });
      
      return {
        sx: Array.from(sx),
        sy: Array.from(sy)
      };
    });
    
    expect(result.sx[0]).toBeCloseTo(2.30, 4);
    expect(result.sx[1]).toBeCloseTo(6.28, 4);
    expect(result.sx[2]).toBeCloseTo(-1.73, 4);
    
    expect(result.sy[0]).toBeCloseTo(3.14, 4);
    expect(result.sy[1]).toBeCloseTo(-2.71, 4);
    expect(result.sy[2]).toBeCloseTo(1.41, 4);
  });

  test('should handle large vectors efficiently', async ({ page }, testInfo) => {
    const { duration, passed } = await page.evaluate(async () => {
      const n = 10000;
      const sx = new Float32Array(n);
      const sy = new Float32Array(n);
      
      for (let i = 0; i < n; i++) {
        sx[i] = i + 1;
        sy[i] = -(i + 1);
      }
      
      const start = performance.now();
      await window.WebGPUBLAS.sswap({ n, sx, sy });
      const end = performance.now();
      
      // Verify first and last few elements
      const firstCorrect = sx[0] === -1 && sy[0] === 1;
      const lastCorrect = sx[n-1] === -n && sy[n-1] === n;
      const middleCorrect = sx[Math.floor(n/2)] === -(Math.floor(n/2) + 1) && 
                           sy[Math.floor(n/2)] === Math.floor(n/2) + 1;
      
      return {
        duration: end - start,
        passed: firstCorrect && lastCorrect && middleCorrect
      };
    });
    
    expect(passed).toBe(true);
    expect(duration).toBeLessThan(500); // Should complete in < 500ms
    
    testInfo.annotations.push({ 
      type: 'performance', 
      description: `10000 elements swapped in ${duration.toFixed(2)}ms` 
    });
  });

  test('should handle edge cases correctly', async ({ page }) => {
    const results = await page.evaluate(async () => {
      // Test with n = 0
      const sx1 = new Float32Array([1, 2, 3]);
      const sy1 = new Float32Array([4, 5, 6]);
      const original1X = Array.from(sx1);
      const original1Y = Array.from(sy1);
      await window.WebGPUBLAS.sswap({ n: 0, sx: sx1, sy: sy1 });
      
      // Test with incx = 0 (should be no-op)
      const sx2 = new Float32Array([1, 2, 3]);
      const sy2 = new Float32Array([4, 5, 6]);
      const original2X = Array.from(sx2);
      const original2Y = Array.from(sy2);
      await window.WebGPUBLAS.sswap({ n: 3, sx: sx2, sy: sy2, incx: 0 });
      
      return {
        case1_unchanged_x: JSON.stringify(Array.from(sx1)) === JSON.stringify(original1X),
        case1_unchanged_y: JSON.stringify(Array.from(sy1)) === JSON.stringify(original1Y),
        case2_unchanged_x: JSON.stringify(Array.from(sx2)) === JSON.stringify(original2X),
        case2_unchanged_y: JSON.stringify(Array.from(sy2)) === JSON.stringify(original2Y),
      };
    });
    
    expect(results.case1_unchanged_x).toBe(true); // n=0 should not modify vectors
    expect(results.case1_unchanged_y).toBe(true);
    expect(results.case2_unchanged_x).toBe(true); // incx=0 should not modify vectors
    expect(results.case2_unchanged_y).toBe(true);
  });

  test('should handle identical vectors correctly', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const sx = new Float32Array([1, 2, 3]);
      const sy = new Float32Array([1, 2, 3]);
      
      await window.WebGPUBLAS.sswap({ n: 3, sx, sy });
      
      return {
        sx: Array.from(sx),
        sy: Array.from(sy)
      };
    });
    
    // Swapping identical vectors should result in the same vectors
    expect(result.sx).toEqual([1, 2, 3]);
    expect(result.sy).toEqual([1, 2, 3]);
  });
});
