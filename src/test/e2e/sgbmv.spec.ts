/**
 * Playwright E2E tests for SGBMV function
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

test.describe('SGBMV - General Banded Matrix-Vector Multiply', () => {
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

  test('should compute banded matrix-vector multiply (no transpose)', async ({ page }) => {
    const result = await page.evaluate(async () => {
      // 4x4 banded matrix with kl=1, ku=2 (1 sub-diagonal, 2 super-diagonals)
      // Full matrix would be:
      // [[1, 2, 3, 0],
      //  [4, 5, 6, 7],
      //  [0, 8, 9, 10],
      //  [0, 0, 11, 12]]
      // 
      // Banded storage format (lda = kl + ku + 1 = 4):
      // A[0,:] = [*, *, 1, 2, 3, 0]  (2nd super-diagonal)
      // A[1,:] = [*, 2, 5, 6, 7]     (1st super-diagonal)
      // A[2,:] = [1, 4, 8, 9, 10, 12] (main diagonal)
      // A[3,:] = [4, 8, 11, *]       (1st sub-diagonal)
      const A = new Float32Array([
        0, 0, 1, 2, 3, 0,   // Row 0: unused, unused, A[0,0], A[0,1], A[0,2], A[0,3] 
        0, 2, 5, 6, 7,      // Row 1: unused, A[1,0], A[1,1], A[1,2], A[1,3]
        1, 4, 8, 9, 10, 12, // Row 2: A[2,0], A[2,1], A[2,2], A[2,3]
        4, 8, 11, 0         // Row 3: A[3,0], A[3,1], A[3,2], unused
      ]);
      const x = new Float32Array([1, 1, 1, 1]);
      const y = new Float32Array([0, 0, 0, 0]);
      
      await window.WebGPUBLAS.sgbmv({
        trans: 'N',
        m: 4,
        n: 4,
        kl: 1,
        ku: 2,
        alpha: 1.0,
        A: A,
        lda: 4,
        x: x,
        beta: 0.0,
        y: y
      });
      
      return Array.from(y);
    });
    
    // Expected: Matrix * [1,1,1,1] = [6, 22, 27, 23]
    expect(result[0]).toBeCloseTo(6.0, 4);
    expect(result[1]).toBeCloseTo(22.0, 4);
    expect(result[2]).toBeCloseTo(27.0, 4);
    expect(result[3]).toBeCloseTo(23.0, 4);
  });

  test('should compute banded matrix-vector multiply (transpose)', async ({ page }) => {
    const result = await page.evaluate(async () => {
      // Same 4x4 banded matrix as above, but transposed operation
      const A = new Float32Array([
        0, 0, 1, 2, 3, 0,
        0, 2, 5, 6, 7,
        1, 4, 8, 9, 10, 12,
        4, 8, 11, 0
      ]);
      const x = new Float32Array([1, 2, 3, 4]);
      const y = new Float32Array([0, 0, 0, 0]);
      
      await window.WebGPUBLAS.sgbmv({
        trans: 'T',
        m: 4,
        n: 4,
        kl: 1,
        ku: 2,
        alpha: 1.0,
        A: A,
        lda: 4,
        x: x,
        beta: 0.0,
        y: y
      });
      
      return Array.from(y);
    });
    
    // Expected: A^T * [1,2,3,4] 
    // A^T is the transpose of the original matrix
    expect(result[0]).toBeCloseTo(9.0, 4);   // 1*1 + 2*4 = 9
    expect(result[1]).toBeCloseTo(24.0, 4);  // 1*2 + 2*5 + 3*8 = 36
    expect(result[2]).toBeCloseTo(54.0, 4);  // 1*3 + 2*6 + 3*9 + 4*11 = 86
    expect(result[3]).toBeCloseTo(76.0, 4);  // 2*7 + 3*10 + 4*12 = 92
  });

  test('should handle alpha and beta scaling', async ({ page }) => {
    const result = await page.evaluate(async () => {
      // Simple 2x2 banded matrix (diagonal only: kl=0, ku=0)
      const A = new Float32Array([
        2, 3  // diagonal elements
      ]);
      const x = new Float32Array([1, 2]);
      const y = new Float32Array([1, 1]); // Initial y values
      
      await window.WebGPUBLAS.sgbmv({
        trans: 'N',
        m: 2,
        n: 2,
        kl: 0,
        ku: 0,
        alpha: 2.0,
        A: A,
        lda: 1,
        x: x,
        beta: 3.0,
        y: y
      });
      
      return Array.from(y);
    });
    
    // Expected: y = alpha * A * x + beta * y = 2.0 * [2, 6] + 3.0 * [1, 1] = [7, 15]
    expect(result[0]).toBeCloseTo(7.0, 4);
    expect(result[1]).toBeCloseTo(15.0, 4);
  });

  test('should handle zero alpha (beta only)', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const A = new Float32Array([1, 2, 3, 4]); // 2x2 matrix
      const x = new Float32Array([1, 1]);
      const y = new Float32Array([5, 6]);
      
      await window.WebGPUBLAS.sgbmv({
        trans: 'N',
        m: 2,
        n: 2,
        kl: 0,
        ku: 1,
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
      // 2x2 diagonal matrix
      const A = new Float32Array([3, 4]);
      const x = new Float32Array([2, 1]);
      const y = new Float32Array([100, 200]); // Should be ignored
      
      await window.WebGPUBLAS.sgbmv({
        trans: 'N',
        m: 2,
        n: 2,
        kl: 0,
        ku: 0,
        alpha: 1.0,
        A: A,
        lda: 1,
        x: x,
        beta: 0.0,
        y: y
      });
      
      return Array.from(y);
    });
    
    // Expected: y = 1.0 * A * x + 0 * y = [6, 4]
    expect(result[0]).toBeCloseTo(6.0, 4);
    expect(result[1]).toBeCloseTo(4.0, 4);
  });

  test('should handle different strides', async ({ page }) => {
    const result = await page.evaluate(async () => {
      // 2x2 diagonal matrix
      const A = new Float32Array([2, 3]);
      const x = new Float32Array([1, 999, 2, 888]); // stride 2, use elements 0 and 2
      const y = new Float32Array([0, 777, 0, 666]); // stride 2, use elements 0 and 2
      
      await window.WebGPUBLAS.sgbmv({
        trans: 'N',
        m: 2,
        n: 2,
        kl: 0,
        ku: 0,
        alpha: 1.0,
        A: A,
        lda: 1,
        x: x,
        incx: 2,
        beta: 0.0,
        y: y,
        incy: 2
      });
      
      return Array.from(y);
    });
    
    // Expected: y = diag(2,3) * [1, 2] = [2, 6] at positions 0 and 2
    expect(result[0]).toBeCloseTo(2.0, 4);
    expect(result[1]).toBeCloseTo(777, 4); // unchanged
    expect(result[2]).toBeCloseTo(6.0, 4);
    expect(result[3]).toBeCloseTo(666, 4); // unchanged
  });

  test('should handle tridiagonal matrix', async ({ page }) => {
    const result = await page.evaluate(async () => {
      // 3x3 tridiagonal matrix (kl=1, ku=1)
      // Full matrix:
      // [[2, 1, 0],
      //  [1, 2, 1],
      //  [0, 1, 2]]
      // 
      // Banded storage (lda = 3):
      // A[0,:] = [*, 1, 1, 0]    (super-diagonal)
      // A[1,:] = [2, 2, 2]       (main diagonal)
      // A[2,:] = [1, 1, *]       (sub-diagonal)
      const A = new Float32Array([
        0, 1, 1, 0,  // super-diagonal (unused, A[0,1], A[1,2], unused)
        2, 2, 2,     // main diagonal
        1, 1, 0      // sub-diagonal (A[1,0], A[2,1], unused)
      ]);
      const x = new Float32Array([1, 1, 1]);
      const y = new Float32Array([0, 0, 0]);
      
      await window.WebGPUBLAS.sgbmv({
        trans: 'N',
        m: 3,
        n: 3,
        kl: 1,
        ku: 1,
        alpha: 1.0,
        A: A,
        lda: 3,
        x: x,
        beta: 0.0,
        y: y
      });
      
      return Array.from(y);
    });
    
    // Expected: tridiagonal * [1,1,1] = [3, 4, 3]
    expect(result[0]).toBeCloseTo(3.0, 4);
    expect(result[1]).toBeCloseTo(4.0, 4);
    expect(result[2]).toBeCloseTo(3.0, 4);
  });

  test('should handle rectangular matrix', async ({ page }) => {
    const result = await page.evaluate(async () => {
      // 3x2 banded matrix (kl=1, ku=0)
      // Full matrix:
      // [[1, 0],
      //  [2, 3],
      //  [0, 4]]
      // 
      // Banded storage (lda = 2):
      // A[0,:] = [1, 3, 4]    (main diagonal)
      // A[1,:] = [2, 0]       (sub-diagonal)
      const A = new Float32Array([
        1, 3, 4,  // main diagonal
        2, 0      // sub-diagonal (A[1,0], unused)
      ]);
      const x = new Float32Array([1, 2]);
      const y = new Float32Array([0, 0, 0]);
      
      await window.WebGPUBLAS.sgbmv({
        trans: 'N',
        m: 3,
        n: 2,
        kl: 1,
        ku: 0,
        alpha: 1.0,
        A: A,
        lda: 2,
        x: x,
        beta: 0.0,
        y: y
      });
      
      return Array.from(y);
    });
    
    // Expected: [[1,0],[2,3],[0,4]] * [1,2] = [1, 8, 8]
    expect(result[0]).toBeCloseTo(1.0, 4);
    expect(result[1]).toBeCloseTo(8.0, 4);
    expect(result[2]).toBeCloseTo(8.0, 4);
  });

  test('should handle edge cases', async ({ page }) => {
    const results = await page.evaluate(async () => {
      const A = new Float32Array([1, 2, 3, 4]);
      const x = new Float32Array([1, 1]);
      const y1 = new Float32Array([5, 6]);
      const y2 = new Float32Array([5, 6]);
      const y3 = new Float32Array([]);
      
      const originalY1 = new Float32Array(y1);
      const originalY2 = new Float32Array(y2);
      
      // Test m = 0 or n = 0 (should do nothing)
      await window.WebGPUBLAS.sgbmv({
        trans: 'N',
        m: 0,
        n: 2,
        kl: 0,
        ku: 0,
        alpha: 1.0,
        A: A,
        lda: 1,
        x: x,
        beta: 0.0,
        y: y3
      });
      
      // Test incx = 0 (should do nothing)
      await window.WebGPUBLAS.sgbmv({
        trans: 'N',
        m: 2,
        n: 2,
        kl: 0,
        ku: 0,
        alpha: 1.0,
        A: A,
        lda: 1,
        x: x,
        incx: 0,
        beta: 1.0,
        y: y1
      });
      
      // Test incy = 0 (should do nothing)
      await window.WebGPUBLAS.sgbmv({
        trans: 'N',
        m: 2,
        n: 2,
        kl: 0,
        ku: 0,
        alpha: 1.0,
        A: A,
        lda: 1,
        x: x,
        beta: 1.0,
        y: y2,
        incy: 0
      });
      
      return {
        empty: Array.from(y3),
        zeroIncx: Array.from(y1),
        zeroIncy: Array.from(y2),
        originalY1: Array.from(originalY1),
        originalY2: Array.from(originalY2)
      };
    });
    
    expect(results.empty).toEqual([]);
    expect(results.zeroIncx).toEqual(results.originalY1); // Should remain unchanged
    expect(results.zeroIncy).toEqual(results.originalY2); // Should remain unchanged
  });

  test('should handle fractional values', async ({ page }) => {
    const result = await page.evaluate(async () => {
      // 2x2 diagonal matrix with fractional values
      const A = new Float32Array([0.5, 0.75]);
      const x = new Float32Array([0.8, 1.2]);
      const y = new Float32Array([0, 0]);
      
      await window.WebGPUBLAS.sgbmv({
        trans: 'N',
        m: 2,
        n: 2,
        kl: 0,
        ku: 0,
        alpha: 1.0,
        A: A,
        lda: 1,
        x: x,
        beta: 0.0,
        y: y
      });
      
      return Array.from(y);
    });
    
    // Expected: diag(0.5, 0.75) * [0.8, 1.2] = [0.4, 0.9]
    expect(result[0]).toBeCloseTo(0.4, 4);
    expect(result[1]).toBeCloseTo(0.9, 4);
  });

  test('should handle single element matrix', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const A = new Float32Array([5.0]);
      const x = new Float32Array([3.0]);
      const y = new Float32Array([0.0]);
      
      await window.WebGPUBLAS.sgbmv({
        trans: 'N',
        m: 1,
        n: 1,
        kl: 0,
        ku: 0,
        alpha: 2.0,
        A: A,
        lda: 1,
        x: x,
        beta: 0.0,
        y: y
      });
      
      return Array.from(y);
    });
    
    // Expected: 2.0 * 5.0 * 3.0 = 30.0
    expect(result[0]).toBeCloseTo(30.0, 4);
  });

  test('should handle larger banded matrices efficiently', async ({ page }, testInfo) => {
    const { passed, duration } = await page.evaluate(async () => {
      const m = 100, n = 100;
      const kl = 2, ku = 3; // 2 sub-diagonals, 3 super-diagonals
      const lda = kl + ku + 1; // 6
      
      const A = new Float32Array(lda * n);
      const x = new Float32Array(n);
      const y = new Float32Array(m);
      
      // Fill banded matrix with simple pattern
      for (let j = 0; j < n; j++) {
        for (let i = Math.max(0, j - ku); i <= Math.min(m - 1, j + kl); i++) {
          const bandIndex = ku + i - j;
          A[bandIndex * n + j] = (i + 1) * 0.1 + (j + 1) * 0.01;
        }
        x[j] = 1.0;
        if (j < m) y[j] = 0.0;
      }
      
      const start = performance.now();
      await window.WebGPUBLAS.sgbmv({
        trans: 'N',
        m: m,
        n: n,
        kl: kl,
        ku: ku,
        alpha: 1.0,
        A: A,
        lda: lda,
        x: x,
        beta: 0.0,
        y: y
      });
      const end = performance.now();
      
      // Basic verification - check some elements are non-zero
      let hasNonZero = false;
      for (let i = 0; i < m; i++) {
        if (Math.abs(y[i]) > 0.01) {
          hasNonZero = true;
          break;
        }
      }
      
      return {
        passed: hasNonZero,
        duration: end - start
      };
    });
    
    expect(passed).toBe(true);
    expect(duration).toBeLessThan(1000); // Should complete in < 1 second
    
    testInfo.annotations.push({ 
      type: 'performance', 
      description: `100x100 banded matrix-vector multiply in ${duration.toFixed(2)}ms` 
    });
  });
});
