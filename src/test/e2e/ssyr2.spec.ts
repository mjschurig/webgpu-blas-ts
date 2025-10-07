/**
 * Playwright E2E tests for SSYR2 function
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

test.describe('SSYR2 - Symmetric Rank-2 Update', () => {
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

  test('should compute symmetric rank-2 update (upper)', async ({ page }) => {
    const result = await page.evaluate(async () => {
      // Start with 2x2 zero matrix
      const A = new Float32Array([
        0, 0,
        0, 0
      ]);
      const x = new Float32Array([1, 2]);
      const y = new Float32Array([3, 4]);
      
      await window.WebGPUBLAS.ssyr2({
        uplo: 'U',
        n: 2,
        alpha: 1.0,
        x: x,
        y: y,
        A: A,
        lda: 2
      });
      
      return Array.from(A);
    });
    
    // Expected: A := alpha * (x * y^T + y * x^T) + A
    // x * y^T = [1, 2] * [3, 4] = [[3, 4], [6, 8]]
    // y * x^T = [3, 4] * [1, 2] = [[3, 6], [4, 8]]
    // Sum = [[6, 10], [10, 16]]
    // Since uplo='U', only upper triangular part is updated:
    // A = [[6, 10], [0, 16]]
    expect(result[0]).toBeCloseTo(6.0, 4);  // A[0,0]
    expect(result[1]).toBeCloseTo(0.0, 4);  // A[1,0] - not updated (upper only)
    expect(result[2]).toBeCloseTo(10.0, 4); // A[0,1]
    expect(result[3]).toBeCloseTo(16.0, 4); // A[1,1]
  });

  test('should compute symmetric rank-2 update (lower)', async ({ page }) => {
    const result = await page.evaluate(async () => {
      // Start with 2x2 zero matrix
      const A = new Float32Array([
        0, 0,
        0, 0
      ]);
      const x = new Float32Array([1, 2]);
      const y = new Float32Array([3, 4]);
      
      await window.WebGPUBLAS.ssyr2({
        uplo: 'L',
        n: 2,
        alpha: 1.0,
        x: x,
        y: y,
        A: A,
        lda: 2
      });
      
      return Array.from(A);
    });
    
    // Expected: A := alpha * (x * y^T + y * x^T) + A
    // Same calculation as above, but lower triangular part is updated:
    // A = [[6, 0], [10, 16]]
    expect(result[0]).toBeCloseTo(6.0, 4);  // A[0,0]
    expect(result[1]).toBeCloseTo(10.0, 4); // A[1,0]
    expect(result[2]).toBeCloseTo(0.0, 4);  // A[0,1] - not updated (lower only)
    expect(result[3]).toBeCloseTo(16.0, 4); // A[1,1]
  });

  test('should handle zero alpha (no operation)', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const A = new Float32Array([
        1, 2,
        3, 4
      ]);
      const originalA = new Float32Array(A);
      const x = new Float32Array([5, 6]);
      const y = new Float32Array([7, 8]);
      
      await window.WebGPUBLAS.ssyr2({
        uplo: 'U',
        n: 2,
        alpha: 0.0,
        x: x,
        y: y,
        A: A,
        lda: 2
      });
      
      return {
        result: Array.from(A),
        original: Array.from(originalA)
      };
    });
    
    // Matrix should remain unchanged when alpha = 0
    expect(result.result).toEqual(result.original);
  });

  test('should handle negative alpha', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const A = new Float32Array([
        1, 0,
        0, 1
      ]);
      const x = new Float32Array([1, 1]);
      const y = new Float32Array([1, 1]);
      
      await window.WebGPUBLAS.ssyr2({
        uplo: 'U',
        n: 2,
        alpha: -1.0,
        x: x,
        y: y,
        A: A,
        lda: 2
      });
      
      return Array.from(A);
    });
    
    // Expected: A = I + (-1) * (x*y^T + y*x^T)
    // x*y^T + y*x^T = [1,1]*[1,1] + [1,1]*[1,1] = [[2,2],[2,2]]
    // A = [[1,0],[0,1]] + (-1)*[[2,2],[2,2]] = [[-1,-2],[-2,-1]]
    // With uplo='U': A = [[-1,-2],[0,-1]]
    expect(result[0]).toBeCloseTo(-1.0, 4); // A[0,0]
    expect(result[1]).toBeCloseTo(0.0, 4);  // A[1,0] - not updated
    expect(result[2]).toBeCloseTo(-2.0, 4); // A[0,1]
    expect(result[3]).toBeCloseTo(-1.0, 4); // A[1,1]
  });

  test('should handle different strides', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const A = new Float32Array([
        0, 0,
        0, 0
      ]);
      const x = new Float32Array([1, 999, 2, 888]); // stride 2, use elements 0 and 2
      const y = new Float32Array([3, 777, 4, 666]); // stride 2, use elements 0 and 2
      
      await window.WebGPUBLAS.ssyr2({
        uplo: 'U',
        n: 2,
        alpha: 1.0,
        x: x,
        incx: 2,
        y: y,
        incy: 2,
        A: A,
        lda: 2
      });
      
      return Array.from(A);
    });
    
    // Using x=[1,2], y=[3,4] with stride 2
    // Same calculation as first test
    expect(result[0]).toBeCloseTo(6.0, 4);  // A[0,0]
    expect(result[1]).toBeCloseTo(0.0, 4);  // A[1,0] 
    expect(result[2]).toBeCloseTo(10.0, 4); // A[0,1]
    expect(result[3]).toBeCloseTo(16.0, 4); // A[1,1]
  });

  test('should handle 3x3 matrix', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const A = new Float32Array([
        1, 0, 0,
        0, 1, 0,
        0, 0, 1
      ]); // 3x3 identity matrix
      const x = new Float32Array([1, 1, 1]);
      const y = new Float32Array([1, 1, 1]);
      
      await window.WebGPUBLAS.ssyr2({
        uplo: 'U',
        n: 3,
        alpha: 0.5,
        x: x,
        y: y,
        A: A,
        lda: 3
      });
      
      return Array.from(A);
    });
    
    // Expected: A = I + 0.5 * (x*y^T + y*x^T)
    // x*y^T + y*x^T = 2 * [[1,1,1],[1,1,1],[1,1,1]]
    // A = I + 0.5 * 2 * ones = I + ones (upper triangular part only)
    // A = [[2,1,1],[0,2,1],[0,0,2]]
    expect(result[0]).toBeCloseTo(2.0, 4); // A[0,0]
    expect(result[1]).toBeCloseTo(0.0, 4); // A[1,0] - not updated
    expect(result[2]).toBeCloseTo(0.0, 4); // A[2,0] - not updated
    expect(result[3]).toBeCloseTo(1.0, 4); // A[0,1]
    expect(result[4]).toBeCloseTo(2.0, 4); // A[1,1]
    expect(result[5]).toBeCloseTo(0.0, 4); // A[2,1] - not updated
    expect(result[6]).toBeCloseTo(1.0, 4); // A[0,2]
    expect(result[7]).toBeCloseTo(1.0, 4); // A[1,2]
    expect(result[8]).toBeCloseTo(2.0, 4); // A[2,2]
  });

  test('should handle fractional values', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const A = new Float32Array([
        0.5, 0,
        0,   0.5
      ]);
      const x = new Float32Array([0.2, 0.3]);
      const y = new Float32Array([0.4, 0.6]);
      
      await window.WebGPUBLAS.ssyr2({
        uplo: 'U',
        n: 2,
        alpha: 2.0,
        x: x,
        y: y,
        A: A,
        lda: 2
      });
      
      return Array.from(A);
    });
    
    // Expected calculation:
    // x*y^T = [0.2,0.3]*[0.4,0.6] = [[0.08,0.12],[0.12,0.18]]
    // y*x^T = [0.4,0.6]*[0.2,0.3] = [[0.08,0.12],[0.12,0.18]]
    // Sum = [[0.16,0.24],[0.24,0.36]]
    // A = [[0.5,0],[0,0.5]] + 2.0*[[0.16,0.24],[0.24,0.36]] (upper only)
    // A = [[0.82,0.48],[0,1.22]]
    expect(result[0]).toBeCloseTo(0.82, 4); // A[0,0]
    expect(result[1]).toBeCloseTo(0.0, 4);  // A[1,0] - not updated
    expect(result[2]).toBeCloseTo(0.48, 4); // A[0,1]
    expect(result[3]).toBeCloseTo(1.22, 4); // A[1,1]
  });

  test('should handle single element matrix', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const A = new Float32Array([5.0]);
      const x = new Float32Array([2.0]);
      const y = new Float32Array([3.0]);
      
      await window.WebGPUBLAS.ssyr2({
        uplo: 'U',
        n: 1,
        alpha: 1.0,
        x: x,
        y: y,
        A: A,
        lda: 1
      });
      
      return Array.from(A);
    });
    
    // Expected: A = 5.0 + 1.0 * (2*3 + 3*2) = 5.0 + 12.0 = 17.0
    expect(result[0]).toBeCloseTo(17.0, 4);
  });

  test('should handle edge cases', async ({ page }) => {
    const results = await page.evaluate(async () => {
      const A1 = new Float32Array([1, 2, 3, 4]);
      const A2 = new Float32Array([1, 2, 3, 4]);
      const A3 = new Float32Array([]);
      const x = new Float32Array([1, 1]);
      const y = new Float32Array([1, 1]);
      
      const originalA1 = new Float32Array(A1);
      const originalA2 = new Float32Array(A2);
      
      // Test n = 0 (should do nothing)
      await window.WebGPUBLAS.ssyr2({
        uplo: 'U',
        n: 0,
        alpha: 1.0,
        x: x,
        y: y,
        A: A3,
        lda: 1
      });
      
      // Test incx = 0 (should do nothing)
      await window.WebGPUBLAS.ssyr2({
        uplo: 'U',
        n: 2,
        alpha: 1.0,
        x: x,
        incx: 0,
        y: y,
        A: A1,
        lda: 2
      });
      
      // Test incy = 0 (should do nothing)
      await window.WebGPUBLAS.ssyr2({
        uplo: 'U',
        n: 2,
        alpha: 1.0,
        x: x,
        y: y,
        incy: 0,
        A: A2,
        lda: 2
      });
      
      return {
        empty: Array.from(A3),
        zeroIncx: Array.from(A1),
        zeroIncy: Array.from(A2),
        originalA1: Array.from(originalA1),
        originalA2: Array.from(originalA2)
      };
    });
    
    expect(results.empty).toEqual([]);
    expect(results.zeroIncx).toEqual(results.originalA1); // Should remain unchanged
    expect(results.zeroIncy).toEqual(results.originalA2); // Should remain unchanged
  });

  test('should handle larger matrices efficiently', async ({ page }, testInfo) => {
    const { passed, duration } = await page.evaluate(async () => {
      const n = 50;
      const A = new Float32Array(n * n);
      const x = new Float32Array(n);
      const y = new Float32Array(n);
      
      // Initialize with small values
      for (let i = 0; i < n; i++) {
        A[i * n + i] = 1.0; // diagonal
        x[i] = 0.1;
        y[i] = 0.1;
      }
      
      const start = performance.now();
      await window.WebGPUBLAS.ssyr2({
        uplo: 'U',
        n: n,
        alpha: 1.0,
        x: x,
        y: y,
        A: A,
        lda: n
      });
      const end = performance.now();
      
      // Basic verification - diagonal elements should be updated
      let allCorrect = true;
      for (let i = 0; i < n; i++) {
        const expected = 1.0 + 2 * 0.1 * 0.1; // 1.0 + 2*x[i]*y[i]
        if (Math.abs(A[i * n + i] - expected) > 0.001) {
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
      description: `50x50 symmetric rank-2 update in ${duration.toFixed(2)}ms` 
    });
  });
});
