/**
 * Playwright E2E tests for SSYMV function
 * Tests run in real browser environment with actual WebGPU shaders
 * Based on FORTRAN BLAS reference tests
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

test.describe('SSYMV - Symmetric Matrix-Vector Multiply', () => {
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

  test('should compute symmetric matrix-vector multiply (upper)', async ({ page }) => {
    const result = await page.evaluate(async () => {
      // 2x2 symmetric matrix (upper triangular stored)
      // A = [[2, 1],
      //      [1, 3]]
      const A = new Float32Array([
        2, 1,  // column 0: [2, 1]
        0, 3   // column 1: [0, 3] (upper triangular, so A[1,0] not used)
      ]);
      const x = new Float32Array([1, 2]);
      const y = new Float32Array([0, 0]);
      
      await window.WebGPUBLAS.ssymv({
        uplo: 'U',
        n: 2,
        alpha: 1.0,
        A: A,
        lda: 2,
        x: x,
        beta: 0.0,
        y: y
      });
      
      return Array.from(y);
    });
    
    // Expected: A * x = [[2, 1], [1, 3]] * [1, 2] = [4, 7]
    expect(result[0]).toBeCloseTo(4.0, 4);
    expect(result[1]).toBeCloseTo(7.0, 4);
  });

  test('should compute symmetric matrix-vector multiply (lower)', async ({ page }) => {
    const result = await page.evaluate(async () => {
      // 2x2 symmetric matrix (lower triangular stored)
      // A = [[2, 1],
      //      [1, 3]]
      const A = new Float32Array([
        2, 1,  // column 0: [2, 1]
        1, 3   // column 1: [1, 3] (lower triangular, so both elements used)
      ]);
      const x = new Float32Array([1, 2]);
      const y = new Float32Array([0, 0]);
      
      await window.WebGPUBLAS.ssymv({
        uplo: 'L',
        n: 2,
        alpha: 1.0,
        A: A,
        lda: 2,
        x: x,
        beta: 0.0,
        y: y
      });
      
      return Array.from(y);
    });
    
    // Expected: A * x = [[2, 1], [1, 3]] * [1, 2] = [4, 7]
    expect(result[0]).toBeCloseTo(4.0, 4);
    expect(result[1]).toBeCloseTo(7.0, 4);
  });

  test('should handle alpha and beta scaling', async ({ page }) => {
    const result = await page.evaluate(async () => {
      // 2x2 identity matrix
      const A = new Float32Array([
        1, 0,
        0, 1
      ]);
      const x = new Float32Array([2, 3]);
      const y = new Float32Array([1, 1]); // Initial y values
      
      await window.WebGPUBLAS.ssymv({
        uplo: 'U',
        n: 2,
        alpha: 2.0,
        A: A,
        lda: 2,
        x: x,
        beta: 3.0,
        y: y
      });
      
      return Array.from(y);
    });
    
    // Expected: y = alpha * A * x + beta * y = 2.0 * [2, 3] + 3.0 * [1, 1] = [7, 9]
    expect(result[0]).toBeCloseTo(7.0, 4);
    expect(result[1]).toBeCloseTo(9.0, 4);
  });

  test('should handle zero alpha (beta only)', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const A = new Float32Array([1, 2, 3, 4]);
      const x = new Float32Array([1, 1]);
      const y = new Float32Array([5, 6]);
      
      await window.WebGPUBLAS.ssymv({
        uplo: 'U',
        n: 2,
        alpha: 0.0,
        A: A,
        lda: 2,
        x: x,
        beta: 2.0,
        y: y
      });
      
      return Array.from(y);
    });
    
    // Expected: y = 0 * A * x + 2.0 * y = [10, 12]
    expect(result[0]).toBeCloseTo(10.0, 4);
    expect(result[1]).toBeCloseTo(12.0, 4);
  });

  test('should handle zero beta (alpha only)', async ({ page }) => {
    const result = await page.evaluate(async () => {
      // 2x2 identity matrix
      const A = new Float32Array([
        1, 0,
        0, 1
      ]);
      const x = new Float32Array([3, 4]);
      const y = new Float32Array([100, 200]); // Should be ignored
      
      await window.WebGPUBLAS.ssymv({
        uplo: 'U',
        n: 2,
        alpha: 1.0,
        A: A,
        lda: 2,
        x: x,
        beta: 0.0,
        y: y
      });
      
      return Array.from(y);
    });
    
    // Expected: y = 1.0 * A * x + 0 * y = [3, 4]
    expect(result[0]).toBeCloseTo(3.0, 4);
    expect(result[1]).toBeCloseTo(4.0, 4);
  });

  test('should handle different strides', async ({ page }) => {
    const result = await page.evaluate(async () => {
      // 2x2 identity matrix
      const A = new Float32Array([
        1, 0,
        0, 1
      ]);
      const x = new Float32Array([1, 999, 2, 888]); // stride 2, use elements 0 and 2
      const y = new Float32Array([0, 777, 0, 666]); // stride 2, use elements 0 and 2
      
      await window.WebGPUBLAS.ssymv({
        uplo: 'U',
        n: 2,
        alpha: 1.0,
        A: A,
        lda: 2,
        x: x,
        incx: 2,
        beta: 0.0,
        y: y,
        incy: 2
      });
      
      return Array.from(y);
    });
    
    // Expected: y = I * [1, 2] = [1, 2] at positions 0 and 2
    expect(result[0]).toBeCloseTo(1.0, 4);
    expect(result[1]).toBeCloseTo(777, 4); // unchanged
    expect(result[2]).toBeCloseTo(2.0, 4);
    expect(result[3]).toBeCloseTo(666, 4); // unchanged
  });

  test('should handle 3x3 matrix', async ({ page }) => {
    const result = await page.evaluate(async () => {
      // 3x3 symmetric matrix (upper triangular)
      // A = [[1, 2, 3],
      //      [2, 4, 5],
      //      [3, 5, 6]]
      const A = new Float32Array([
        1, 2, 3,  // column 0
        0, 4, 5,  // column 1 (upper triangular)
        0, 0, 6   // column 2 (upper triangular)
      ]);
      const x = new Float32Array([1, 1, 1]);
      const y = new Float32Array([0, 0, 0]);
      
      await window.WebGPUBLAS.ssymv({
        uplo: 'U',
        n: 3,
        alpha: 1.0,
        A: A,
        lda: 3,
        x: x,
        beta: 0.0,
        y: y
      });
      
      return Array.from(y);
    });
    
    // Expected: A * [1, 1, 1] = [6, 11, 14]
    expect(result[0]).toBeCloseTo(6.0, 4);
    expect(result[1]).toBeCloseTo(11.0, 4);
    expect(result[2]).toBeCloseTo(14.0, 4);
  });

  test('should handle edge cases', async ({ page }) => {
    const results = await page.evaluate(async () => {
      const A = new Float32Array([1]);
      const x = new Float32Array([5]);
      const y1 = new Float32Array([0]);
      const y2 = new Float32Array([0]);
      const y3 = new Float32Array([]);
      
      // Test n = 0 (should do nothing)
      await window.WebGPUBLAS.ssymv({
        uplo: 'U',
        n: 0,
        alpha: 1.0,
        A: A,
        lda: 1,
        x: x,
        beta: 0.0,
        y: y3
      });
      
      // Test 1x1 matrix
      await window.WebGPUBLAS.ssymv({
        uplo: 'U',
        n: 1,
        alpha: 2.0,
        A: A,
        lda: 1,
        x: x,
        beta: 0.0,
        y: y1
      });
      
      // Test with incx = 0 (should do nothing)
      await window.WebGPUBLAS.ssymv({
        uplo: 'U',
        n: 1,
        alpha: 1.0,
        A: A,
        lda: 1,
        x: x,
        incx: 0,
        beta: 0.0,
        y: y2
      });
      
      return {
        empty: Array.from(y3),
        single: Array.from(y1),
        zeroIncx: Array.from(y2)
      };
    });
    
    expect(results.empty).toEqual([]);
    expect(results.single[0]).toBeCloseTo(10.0, 4); // 2.0 * 1 * 5 = 10
    expect(results.zeroIncx[0]).toBeCloseTo(0.0, 4); // Should remain unchanged
  });

  test('should handle fractional values', async ({ page }) => {
    const result = await page.evaluate(async () => {
      // 2x2 matrix with fractional values
      const A = new Float32Array([
        0.5, 0.25,
        0,   0.75
      ]);
      const x = new Float32Array([0.8, 1.2]);
      const y = new Float32Array([0, 0]);
      
      await window.WebGPUBLAS.ssymv({
        uplo: 'U',
        n: 2,
        alpha: 1.0,
        A: A,
        lda: 2,
        x: x,
        beta: 0.0,
        y: y
      });
      
      return Array.from(y);
    });
    
    // Expected calculation:
    // A = [[0.5, 0.25], [0.25, 0.75]] (symmetric)
    // A * [0.8, 1.2] = [0.5*0.8 + 0.25*1.2, 0.25*0.8 + 0.75*1.2] = [0.7, 1.1]
    expect(result[0]).toBeCloseTo(0.7, 4);
    expect(result[1]).toBeCloseTo(1.1, 4);
  });

  test('should handle larger matrices efficiently', async ({ page }, testInfo) => {
    const { passed, duration } = await page.evaluate(async () => {
      const n = 100;
      const A = new Float32Array(n * n);
      const x = new Float32Array(n);
      const y = new Float32Array(n);
      
      // Create a symmetric matrix (identity + small perturbation)
      for (let i = 0; i < n; i++) {
        A[i * n + i] = 1.0; // diagonal
        x[i] = 1.0; // all ones
        y[i] = 0.0;
        for (let j = i + 1; j < n; j++) {
          A[j * n + i] = 0.01; // small off-diagonal (lower triangle, but using upper)
        }
      }
      
      const start = performance.now();
      await window.WebGPUBLAS.ssymv({
        uplo: 'U',
        n: n,
        alpha: 1.0,
        A: A,
        lda: n,
        x: x,
        beta: 0.0,
        y: y
      });
      const end = performance.now();
      
      // Verify result - diagonal elements should give 1.0, off-diagonal adds 0.01*(n-1-i)
      let allCorrect = true;
      for (let i = 0; i < n; i++) {
        const expected = 1.0 + 0.01 * (n - 1 - i);
        if (Math.abs(y[i] - expected) > 0.01) {
          allCorrect = false;
          break;
        }
      }
      
      return {
        passed: allCorrect,
        duration: end - start
      };
    });
    
    expect(passed).toBe(true);
    expect(duration).toBeLessThan(1000); // Should complete in < 1 second
    
    testInfo.annotations.push({ 
      type: 'performance', 
      description: `100x100 symmetric matrix-vector multiply in ${duration.toFixed(2)}ms` 
    });
  });
});
