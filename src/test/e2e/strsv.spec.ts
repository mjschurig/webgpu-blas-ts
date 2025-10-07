/**
 * Playwright E2E tests for STRSV function
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

test.describe('STRSV - Triangular Solve', () => {
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

  test('should solve upper triangular system (no transpose)', async ({ page }) => {
    const result = await page.evaluate(async () => {
      // 3x3 upper triangular matrix
      // A = [[2, 1, 3],
      //      [0, 1, 2],
      //      [0, 0, 1]]
      const A = new Float32Array([
        2, 0, 0,  // column 0
        1, 1, 0,  // column 1
        3, 2, 1   // column 2
      ]);
      const x = new Float32Array([8, 5, 2]); // Right-hand side b
      
      await window.WebGPUBLAS.strsv({
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
    
    // Solving: A * x = b
    // [[2,1,3],[0,1,2],[0,0,1]] * x = [8,5,2]
    // From bottom up: x[2] = 2/1 = 2
    //                 x[1] = (5 - 2*2)/1 = 1  
    //                 x[0] = (8 - 1*1 - 3*2)/2 = 1/2 = 0.5
    expect(result[0]).toBeCloseTo(0.5, 4);
    expect(result[1]).toBeCloseTo(1.0, 4);
    expect(result[2]).toBeCloseTo(2.0, 4);
  });

  test('should solve lower triangular system (no transpose)', async ({ page }) => {
    const result = await page.evaluate(async () => {
      // 3x3 lower triangular matrix
      // A = [[2, 0, 0],
      //      [1, 1, 0],
      //      [3, 2, 1]]
      const A = new Float32Array([
        2, 1, 3,  // column 0
        0, 1, 2,  // column 1
        0, 0, 1   // column 2
      ]);
      const x = new Float32Array([1, 2, 5]); // Right-hand side b
      
      await window.WebGPUBLAS.strsv({
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
    
    // Solving: A * x = b
    // [[2,0,0],[1,1,0],[3,2,1]] * x = [1,2,5]
    // From top down: x[0] = 1/2 = 0.5
    //                x[1] = (2 - 1*0.5)/1 = 1.5
    //                x[2] = (5 - 3*0.5 - 2*1.5)/1 = 2
    expect(result[0]).toBeCloseTo(0.5, 4);
    expect(result[1]).toBeCloseTo(1.5, 4);
    expect(result[2]).toBeCloseTo(2.0, 4);
  });

  test('should solve upper triangular system (transpose)', async ({ page }) => {
    const result = await page.evaluate(async () => {
      // 3x3 upper triangular matrix
      // A = [[2, 1, 3],
      //      [0, 1, 2],
      //      [0, 0, 1]]
      // A^T = [[2, 0, 0],
      //        [1, 1, 0],
      //        [3, 2, 1]]
      const A = new Float32Array([
        2, 0, 0,  // column 0
        1, 1, 0,  // column 1
        3, 2, 1   // column 2
      ]);
      const x = new Float32Array([1, 2, 5]); // Right-hand side b
      
      await window.WebGPUBLAS.strsv({
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
    
    // Solving: A^T * x = b where A^T is lower triangular
    // Same as previous test result
    expect(result[0]).toBeCloseTo(0.5, 4);
    expect(result[1]).toBeCloseTo(1.5, 4);
    expect(result[2]).toBeCloseTo(2.0, 4);
  });

  test('should solve system with identity matrix', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const A = new Float32Array([
        1, 0, 0,
        0, 1, 0,
        0, 0, 1
      ]);
      const x = new Float32Array([5, 7, 9]);
      
      await window.WebGPUBLAS.strsv({
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
    
    // Identity matrix: solution should be the same as input
    expect(result[0]).toBeCloseTo(5.0, 4);
    expect(result[1]).toBeCloseTo(7.0, 4);
    expect(result[2]).toBeCloseTo(9.0, 4);
  });

  test('should handle unit diagonal', async ({ page }) => {
    const result = await page.evaluate(async () => {
      // 3x3 upper triangular matrix with unit diagonal
      // Diagonal elements are assumed to be 1, regardless of stored values
      const A = new Float32Array([
        999, 0, 0,  // column 0 (diagonal element ignored)
        2,   888, 0,  // column 1 (diagonal element ignored)
        3,   4, 777   // column 2 (diagonal element ignored)
      ]);
      const x = new Float32Array([4, 6, 3]); // Right-hand side b
      
      await window.WebGPUBLAS.strsv({
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
    
    // Solving: A * x = b with unit diagonal
    // [[1,2,3],[0,1,4],[0,0,1]] * x = [4,6,3]
    // From bottom up: x[2] = 3/1 = 3
    //                 x[1] = (6 - 4*3)/1 = -6  
    //                 x[0] = (4 - 2*(-6) - 3*3)/1 = 7
    expect(result[0]).toBeCloseTo(7.0, 4);
    expect(result[1]).toBeCloseTo(-6.0, 4);
    expect(result[2]).toBeCloseTo(3.0, 4);
  });

  test('should handle stride correctly', async ({ page }) => {
    const result = await page.evaluate(async () => {
      // 2x2 upper triangular matrix
      const A = new Float32Array([
        2, 0,
        1, 3
      ]);
      const x = new Float32Array([6, 999, 9, 888]); // stride 2, use elements 0 and 2
      
      await window.WebGPUBLAS.strsv({
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
    
    // Solving: [[2,1],[0,3]] * [x0,x1] = [6,9]
    // x[1] = 9/3 = 3, x[0] = (6-1*3)/2 = 1.5
    // Results should be at positions 0 and 2
    expect(result[0]).toBeCloseTo(1.5, 4);
    expect(result[1]).toBeCloseTo(999, 4); // unchanged
    expect(result[2]).toBeCloseTo(3.0, 4);
    expect(result[3]).toBeCloseTo(888, 4); // unchanged
  });

  test('should handle single element system', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const A = new Float32Array([4.0]);
      const x = new Float32Array([12.0]);
      
      await window.WebGPUBLAS.strsv({
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
    
    // Solving: 4 * x = 12, so x = 3
    expect(result[0]).toBeCloseTo(3.0, 4);
  });

  test('should handle single element system with unit diagonal', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const A = new Float32Array([999.0]); // Value should be ignored
      const x = new Float32Array([7.0]);
      
      await window.WebGPUBLAS.strsv({
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
    
    // Solving: 1 * x = 7, so x = 7 (unit diagonal)
    expect(result[0]).toBeCloseTo(7.0, 4);
  });

  test('should handle fractional values', async ({ page }) => {
    const result = await page.evaluate(async () => {
      // 2x2 upper triangular matrix with fractional values
      const A = new Float32Array([
        0.5, 0,
        0.25, 0.75
      ]);
      const x = new Float32Array([1.0, 1.5]);
      
      await window.WebGPUBLAS.strsv({
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
    
    // Solving: [[0.5,0.25],[0,0.75]] * x = [1.0,1.5]
    // x[1] = 1.5/0.75 = 2
    // x[0] = (1.0 - 0.25*2)/0.5 = 1
    expect(result[0]).toBeCloseTo(1.0, 4);
    expect(result[1]).toBeCloseTo(2.0, 4);
  });

  test('should handle edge cases', async ({ page }) => {
    const results = await page.evaluate(async () => {
      const A = new Float32Array([1, 2, 3, 4]);
      const x1 = new Float32Array([5, 6]);
      const x2 = new Float32Array([]);
      
      const originalX1 = new Float32Array(x1);
      
      // Test n = 0 (should do nothing)
      await window.WebGPUBLAS.strsv({
        uplo: 'U',
        trans: 'N',
        diag: 'N',
        n: 0,
        A: A,
        lda: 1,
        x: x2
      });
      
      // Test incx = 0 (should do nothing)
      await window.WebGPUBLAS.strsv({
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
        empty: Array.from(x2),
        zeroIncx: Array.from(x1),
        originalX1: Array.from(originalX1)
      };
    });
    
    expect(results.empty).toEqual([]);
    expect(results.zeroIncx).toEqual(results.originalX1); // Should remain unchanged
  });

  test('should solve well-conditioned system', async ({ page }) => {
    const result = await page.evaluate(async () => {
      // Well-conditioned 3x3 upper triangular system
      const A = new Float32Array([
        1, 0, 0,  // column 0
        0, 2, 0,  // column 1
        0, 0, 3   // column 2
      ]);
      const x = new Float32Array([1, 4, 9]); // b vector
      
      await window.WebGPUBLAS.strsv({
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
    
    // Diagonal system: x[i] = b[i] / A[i][i]
    expect(result[0]).toBeCloseTo(1.0, 4); // 1/1
    expect(result[1]).toBeCloseTo(2.0, 4); // 4/2
    expect(result[2]).toBeCloseTo(3.0, 4); // 9/3
  });

  test('should handle larger system', async ({ page }, testInfo) => {
    const { passed, duration } = await page.evaluate(async () => {
      const n = 50;
      const A = new Float32Array(n * n);
      const x = new Float32Array(n);
      
      // Create upper triangular matrix with known solution
      for (let i = 0; i < n; i++) {
        A[i * n + i] = i + 1; // diagonal: 1, 2, 3, ..., n
        x[i] = (i + 1) * (i + 1); // b vector that gives solution [1, 2, 3, ..., n]
        
        // Small upper triangular values
        for (let j = i + 1; j < n; j++) {
          A[j * n + i] = 0.01;
          x[i] += 0.01 * (j + 1); // adjust b for upper triangular part
        }
      }
      
      const start = performance.now();
      await window.WebGPUBLAS.strsv({
        uplo: 'U',
        trans: 'N',
        diag: 'N',
        n: n,
        A: A,
        lda: n,
        x: x
      });
      const end = performance.now();
      
      // Verify solution is approximately [1, 2, 3, ..., n]
      let allCorrect = true;
      for (let i = 0; i < n; i++) {
        const expected = i + 1;
        if (Math.abs(x[i] - expected) > 0.1) {
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
    expect(duration).toBeLessThan(2000); // Triangular solve can be slower due to dependencies
    
    testInfo.annotations.push({ 
      type: 'performance', 
      description: `50x50 triangular solve in ${duration.toFixed(2)}ms` 
    });
  });

  test('should handle complex combination (lower, transpose, unit)', async ({ page }) => {
    const result = await page.evaluate(async () => {
      // 3x3 lower triangular matrix with unit diagonal
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
      const x = new Float32Array([6, 5, 1]); // Right-hand side b
      
      await window.WebGPUBLAS.strsv({
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
    
    // Solving: A^T * x = b where A^T = [[1,2,3],[0,1,4],[0,0,1]]
    // From bottom up: x[2] = 1/1 = 1
    //                 x[1] = (5 - 4*1)/1 = 1  
    //                 x[0] = (6 - 2*1 - 3*1)/1 = 1
    expect(result[0]).toBeCloseTo(1.0, 4);
    expect(result[1]).toBeCloseTo(1.0, 4);
    expect(result[2]).toBeCloseTo(1.0, 4);
  });
});
