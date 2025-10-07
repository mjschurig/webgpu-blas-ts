/**
 * Playwright E2E tests for STRMV function
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

test.describe('STRMV - Triangular Matrix-Vector Multiply', () => {
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

  test('should compute upper triangular matrix-vector multiply (no transpose)', async ({ page }) => {
    const result = await page.evaluate(async () => {
      // 3x3 upper triangular matrix
      // A = [[2, 3, 4],
      //      [0, 1, 5],
      //      [0, 0, 6]]
      const A = new Float32Array([
        2, 0, 0,  // column 0
        3, 1, 0,  // column 1
        4, 5, 6   // column 2
      ]);
      const x = new Float32Array([1, 2, 3]);
      
      await window.WebGPUBLAS.strmv({
        uplo: 'U',
        trans: 'N',
        diag: 'N',
        n: 3,
        A: A,
        lda: 3,
        x: x
      });
      
      return Array.from(x);
    });
    
    // Expected: A * x = [[2,3,4],[0,1,5],[0,0,6]] * [1,2,3] = [20, 17, 18]
    expect(result[0]).toBeCloseTo(20.0, 4);
    expect(result[1]).toBeCloseTo(17.0, 4);
    expect(result[2]).toBeCloseTo(18.0, 4);
  });

  test('should compute lower triangular matrix-vector multiply (no transpose)', async ({ page }) => {
    const result = await page.evaluate(async () => {
      // 3x3 lower triangular matrix
      // A = [[2, 0, 0],
      //      [3, 1, 0],
      //      [4, 5, 6]]
      const A = new Float32Array([
        2, 3, 4,  // column 0
        0, 1, 5,  // column 1
        0, 0, 6   // column 2
      ]);
      const x = new Float32Array([1, 2, 3]);
      
      await window.WebGPUBLAS.strmv({
        uplo: 'L',
        trans: 'N',
        diag: 'N',
        n: 3,
        A: A,
        lda: 3,
        x: x
      });
      
      return Array.from(x);
    });
    
    // Expected: A * x = [[2,0,0],[3,1,0],[4,5,6]] * [1,2,3] = [2, 5, 32]
    expect(result[0]).toBeCloseTo(2.0, 4);
    expect(result[1]).toBeCloseTo(5.0, 4);
    expect(result[2]).toBeCloseTo(32.0, 4);
  });

  test('should compute upper triangular matrix-vector multiply (transpose)', async ({ page }) => {
    const result = await page.evaluate(async () => {
      // 3x3 upper triangular matrix
      // A = [[2, 3, 4],
      //      [0, 1, 5],
      //      [0, 0, 6]]
      // A^T = [[2, 0, 0],
      //        [3, 1, 0],
      //        [4, 5, 6]]
      const A = new Float32Array([
        2, 0, 0,  // column 0
        3, 1, 0,  // column 1
        4, 5, 6   // column 2
      ]);
      const x = new Float32Array([1, 2, 3]);
      
      await window.WebGPUBLAS.strmv({
        uplo: 'U',
        trans: 'T',
        diag: 'N',
        n: 3,
        A: A,
        lda: 3,
        x: x
      });
      
      return Array.from(x);
    });
    
    // Expected: A^T * x = [[2,0,0],[3,1,0],[4,5,6]] * [1,2,3] = [2, 5, 32]
    expect(result[0]).toBeCloseTo(2.0, 4);
    expect(result[1]).toBeCloseTo(5.0, 4);
    expect(result[2]).toBeCloseTo(32.0, 4);
  });

  test('should handle unit diagonal (upper)', async ({ page }) => {
    const result = await page.evaluate(async () => {
      // 3x3 upper triangular matrix with unit diagonal
      // Diagonal elements are assumed to be 1, regardless of stored values
      const A = new Float32Array([
        999, 0, 0,  // column 0 (diagonal element ignored)
        2,   888, 0,  // column 1 (diagonal element ignored)
        3,   4, 777   // column 2 (diagonal element ignored)
      ]);
      const x = new Float32Array([1, 2, 3]);
      
      await window.WebGPUBLAS.strmv({
        uplo: 'U',
        trans: 'N',
        diag: 'U', // Unit diagonal
        n: 3,
        A: A,
        lda: 3,
        x: x
      });
      
      return Array.from(x);
    });
    
    // Expected: A * x with unit diagonal = [[1,2,3],[0,1,4],[0,0,1]] * [1,2,3] = [14, 14, 3]
    expect(result[0]).toBeCloseTo(14.0, 4);
    expect(result[1]).toBeCloseTo(14.0, 4);
    expect(result[2]).toBeCloseTo(3.0, 4);
  });

  test('should handle stride correctly', async ({ page }) => {
    const result = await page.evaluate(async () => {
      // 2x2 upper triangular matrix
      const A = new Float32Array([
        2, 0,
        3, 4
      ]);
      const x = new Float32Array([1, 999, 2, 888]); // stride 2, use elements 0 and 2
      
      await window.WebGPUBLAS.strmv({
        uplo: 'U',
        trans: 'N',
        diag: 'N',
        n: 2,
        A: A,
        lda: 2,
        x: x,
        incx: 2
      });
      
      return Array.from(x);
    });
    
    // Expected: A * [1, 2] = [[2,3],[0,4]] * [1,2] = [8, 8]
    // Results should be at positions 0 and 2
    expect(result[0]).toBeCloseTo(8.0, 4);
    expect(result[1]).toBeCloseTo(999, 4); // unchanged
    expect(result[2]).toBeCloseTo(8.0, 4);
    expect(result[3]).toBeCloseTo(888, 4); // unchanged
  });

  test('should handle 2x2 identity matrix', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const A = new Float32Array([
        1, 0,
        0, 1
      ]);
      const x = new Float32Array([5, 7]);
      
      await window.WebGPUBLAS.strmv({
        uplo: 'U',
        trans: 'N',
        diag: 'N',
        n: 2,
        A: A,
        lda: 2,
        x: x
      });
      
      return Array.from(x);
    });
    
    // Expected: I * x = x
    expect(result[0]).toBeCloseTo(5.0, 4);
    expect(result[1]).toBeCloseTo(7.0, 4);
  });

  test('should handle fractional values', async ({ page }) => {
    const result = await page.evaluate(async () => {
      // 2x2 upper triangular matrix with fractional values
      const A = new Float32Array([
        0.5, 0,
        0.25, 0.75
      ]);
      const x = new Float32Array([2.0, 4.0]);
      
      await window.WebGPUBLAS.strmv({
        uplo: 'U',
        trans: 'N',
        diag: 'N',
        n: 2,
        A: A,
        lda: 2,
        x: x
      });
      
      return Array.from(x);
    });
    
    // Expected: A * x = [[0.5,0.25],[0,0.75]] * [2,4] = [2, 3]
    expect(result[0]).toBeCloseTo(2.0, 4);
    expect(result[1]).toBeCloseTo(3.0, 4);
  });

  test('should handle single element matrix', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const A = new Float32Array([3.0]);
      const x = new Float32Array([4.0]);
      
      await window.WebGPUBLAS.strmv({
        uplo: 'U',
        trans: 'N',
        diag: 'N',
        n: 1,
        A: A,
        lda: 1,
        x: x
      });
      
      return Array.from(x);
    });
    
    // Expected: 3 * 4 = 12
    expect(result[0]).toBeCloseTo(12.0, 4);
  });

  test('should handle single element matrix with unit diagonal', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const A = new Float32Array([999.0]); // Value should be ignored
      const x = new Float32Array([4.0]);
      
      await window.WebGPUBLAS.strmv({
        uplo: 'U',
        trans: 'N',
        diag: 'U', // Unit diagonal
        n: 1,
        A: A,
        lda: 1,
        x: x
      });
      
      return Array.from(x);
    });
    
    // Expected: 1 * 4 = 4 (unit diagonal)
    expect(result[0]).toBeCloseTo(4.0, 4);
  });

  test('should handle edge cases', async ({ page }) => {
    const results = await page.evaluate(async () => {
      const A = new Float32Array([1, 2, 3, 4]);
      const x1 = new Float32Array([5, 6]);
      const x2 = new Float32Array([5, 6]);
      const x3 = new Float32Array([]);
      
      const originalX1 = new Float32Array(x1);
      const originalX2 = new Float32Array(x2);
      
      // Test n = 0 (should do nothing)
      await window.WebGPUBLAS.strmv({
        uplo: 'U',
        trans: 'N',
        diag: 'N',
        n: 0,
        A: A,
        lda: 1,
        x: x3
      });
      
      // Test incx = 0 (should do nothing)
      await window.WebGPUBLAS.strmv({
        uplo: 'U',
        trans: 'N',
        diag: 'N',
        n: 2,
        A: A,
        lda: 2,
        x: x1,
        incx: 0
      });
      
      return {
        empty: Array.from(x3),
        zeroIncx: Array.from(x1),
        originalX1: Array.from(originalX1)
      };
    });
    
    expect(results.empty).toEqual([]);
    expect(results.zeroIncx).toEqual(results.originalX1); // Should remain unchanged
  });

  test('should handle larger matrices efficiently', async ({ page }, testInfo) => {
    const { passed, duration } = await page.evaluate(async () => {
      const n = 100;
      const A = new Float32Array(n * n);
      const x = new Float32Array(n);
      
      // Create upper triangular matrix (identity + small upper triangle)
      for (let i = 0; i < n; i++) {
        A[i * n + i] = 1.0; // diagonal
        x[i] = 1.0;
        for (let j = i + 1; j < n; j++) {
          A[j * n + i] = 0.01; // small upper triangular values
        }
      }
      
      const start = performance.now();
      await window.WebGPUBLAS.strmv({
        uplo: 'U',
        trans: 'N',
        diag: 'N',
        n: n,
        A: A,
        lda: n,
        x: x
      });
      const end = performance.now();
      
      // Verify result - each element should be close to expected value
      let allCorrect = true;
      for (let i = 0; i < n; i++) {
        const expected = 1.0 + 0.01 * (n - 1 - i); // diagonal + upper triangle contribution
        if (Math.abs(x[i] - expected) > 0.01) {
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
      description: `100x100 triangular matrix-vector multiply in ${duration.toFixed(2)}ms` 
    });
  });

  test('should handle complex combination (lower, transpose, unit)', async ({ page }) => {
    const result = await page.evaluate(async () => {
      // 3x3 lower triangular matrix
      // A = [[1, 0, 0],   (unit diagonal, so diagonal values ignored)
      //      [2, 1, 0],
      //      [3, 4, 1]]
      // A^T = [[1, 2, 3],
      //        [0, 1, 4],
      //        [0, 0, 1]]
      const A = new Float32Array([
        999, 2, 3,  // column 0 (diagonal ignored due to unit)
        0, 888, 4,  // column 1 (diagonal ignored due to unit)
        0, 0, 777   // column 2 (diagonal ignored due to unit)
      ]);
      const x = new Float32Array([1, 1, 1]);
      
      await window.WebGPUBLAS.strmv({
        uplo: 'L',
        trans: 'T',
        diag: 'U', // Unit diagonal
        n: 3,
        A: A,
        lda: 3,
        x: x
      });
      
      return Array.from(x);
    });
    
    // Expected: A^T * x = [[1,2,3],[0,1,4],[0,0,1]] * [1,1,1] = [6, 5, 1]
    expect(result[0]).toBeCloseTo(6.0, 4);
    expect(result[1]).toBeCloseTo(5.0, 4);
    expect(result[2]).toBeCloseTo(1.0, 4);
  });
});
