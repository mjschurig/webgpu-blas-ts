/**
 * End-to-end tests for SSYR (Symmetric Rank-1 Update)
 * 
 * Tests the WebGPU implementation against known results and reference implementations.
 */

import { test, expect } from '@playwright/test';

declare global {
  interface Window {
    WebGPUBLAS: any;
  }
}

test.describe('SSYR', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('http://localhost:3000/test.html');
    await page.waitForFunction(() => window.WebGPUBLAS?.ssyr);
    // Initialize WebGPU context
    await page.evaluate(async () => {
      await window.WebGPUBLAS.getWebGPUContext();
    });
  });

  test('should compute symmetric rank-1 update (upper)', async ({ page }) => {
    const result = await page.evaluate(async () => {
      // 2x2 symmetric matrix A (upper), x = [1, 2], alpha = 1.0
      // A = [[1, 3], [3, 4]] stored as column-major: [1, 3, 3, 4]
      // x * x^T = [[1*1, 1*2], [2*1, 2*2]] = [[1, 2], [2, 4]]
      // Expected upper part: A = [[1, 3], [_, 4]] + 1 * [[1, 2], [_, 4]] = [[2, 5], [_, 8]]
      const A = new Float32Array([1, 3, 3, 4]); // column-major
      const x = new Float32Array([1, 2]);
      
      await window.WebGPUBLAS.ssyr({
        uplo: 'U',
        n: 2,
        alpha: 1.0,
        x: x,
        incx: 1,
        A: A,
        lda: 2
      });
      
      return Array.from(A);
    });
    
    // Expected upper triangular: [2, 3, 5, 8] (only upper part modified)
    expect(result[0]).toBeCloseTo(2, 5); // A[0,0] = 1 + 1*1 = 2
    expect(result[1]).toBeCloseTo(3, 5); // A[1,0] = 3 (unchanged, lower part)
    expect(result[2]).toBeCloseTo(5, 5); // A[0,1] = 3 + 1*2 = 5
    expect(result[3]).toBeCloseTo(8, 5); // A[1,1] = 4 + 2*2 = 8
  });

  test('should compute symmetric rank-1 update (lower)', async ({ page }) => {
    const result = await page.evaluate(async () => {
      // 2x2 symmetric matrix A (lower), x = [1, 2], alpha = 1.0
      // A = [[1, 3], [3, 4]] stored as column-major: [1, 3, 3, 4]
      // Expected lower part: A = [[1, _], [3, 4]] + 1 * [[1, _], [2, 4]] = [[2, _], [5, 8]]
      const A = new Float32Array([1, 3, 3, 4]); // column-major
      const x = new Float32Array([1, 2]);
      
      await window.WebGPUBLAS.ssyr({
        uplo: 'L',
        n: 2,
        alpha: 1.0,
        x: x,
        incx: 1,
        A: A,
        lda: 2
      });
      
      return Array.from(A);
    });
    
    // Expected lower triangular: [2, 5, 3, 8] (only lower part modified)
    expect(result[0]).toBeCloseTo(2, 5); // A[0,0] = 1 + 1*1 = 2
    expect(result[1]).toBeCloseTo(5, 5); // A[1,0] = 3 + 2*1 = 5
    expect(result[2]).toBeCloseTo(3, 5); // A[0,1] = 3 (unchanged, upper part)
    expect(result[3]).toBeCloseTo(8, 5); // A[1,1] = 4 + 2*2 = 8
  });

  test('should handle zero alpha (no operation)', async ({ page }) => {
    const result = await page.evaluate(async () => {
      // Test alpha = 0, matrix should remain unchanged
      const A = new Float32Array([1, 2, 3, 4]);
      const x = new Float32Array([5, 6]);
      
      await window.WebGPUBLAS.ssyr({
        uplo: 'U',
        n: 2,
        alpha: 0.0,
        x: x,
        incx: 1,
        A: A,
        lda: 2
      });
      
      return Array.from(A);
    });
    
    // Should remain unchanged
    expect(result[0]).toBeCloseTo(1, 5);
    expect(result[1]).toBeCloseTo(2, 5);
    expect(result[2]).toBeCloseTo(3, 5);
    expect(result[3]).toBeCloseTo(4, 5);
  });

  test('should handle negative alpha', async ({ page }) => {
    const result = await page.evaluate(async () => {
      // Test with alpha = -1.0
      // A = [[2, 0], [0, 2]] (2*I), x = [1, 1]
      // x * x^T = [[1*1, 1*1], [1*1, 1*1]] = [[1, 1], [1, 1]]
      // Expected upper: A = [[2, 0], [_, 2]] + (-1) * [[1, 1], [_, 1]] = [[1, -1], [_, 1]]
      const A = new Float32Array([2, 0, 0, 2]); // 2*I matrix, column-major
      const x = new Float32Array([1, 1]);
      
      await window.WebGPUBLAS.ssyr({
        uplo: 'U',
        n: 2,
        alpha: -1.0,
        x: x,
        incx: 1,
        A: A,
        lda: 2
      });
      
      return Array.from(A);
    });
    
    // Expected: [1, 0, -1, 1] (upper part modified)
    expect(result[0]).toBeCloseTo(1, 5);  // A[0,0] = 2 - 1*1 = 1
    expect(result[1]).toBeCloseTo(0, 5);  // A[1,0] = 0 (unchanged)
    expect(result[2]).toBeCloseTo(-1, 5); // A[0,1] = 0 - 1*1 = -1
    expect(result[3]).toBeCloseTo(1, 5);  // A[1,1] = 2 - 1*1 = 1
  });

  test('should handle 3x3 matrix', async ({ page }) => {
    const result = await page.evaluate(async () => {
      // 3x3 identity matrix, x = [1, 1, 1], alpha = 2.0, upper triangular
      // I + 2 * [1,1,1] * [1,1,1]^T = I + 2 * [[1,1,1],[1,1,1],[1,1,1]]
      // Expected upper: [[1+2, 0+2, 0+2], [_, 1+2, 0+2], [_, _, 1+2]] = [[3, 2, 2], [_, 3, 2], [_, _, 3]]
      const A = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]); // 3x3 identity, column-major
      const x = new Float32Array([1, 1, 1]);
      
      await window.WebGPUBLAS.ssyr({
        uplo: 'U',
        n: 3,
        alpha: 2.0,
        x: x,
        incx: 1,
        A: A,
        lda: 3
      });
      
      return Array.from(A);
    });
    
    // Expected: [3, 0, 0, 2, 3, 0, 2, 2, 3] (upper part updated)
    expect(result[0]).toBeCloseTo(3, 5); // A[0,0] = 1 + 2*1*1 = 3
    expect(result[1]).toBeCloseTo(0, 5); // A[1,0] = 0 (unchanged)
    expect(result[2]).toBeCloseTo(0, 5); // A[2,0] = 0 (unchanged)
    expect(result[3]).toBeCloseTo(2, 5); // A[0,1] = 0 + 2*1*1 = 2
    expect(result[4]).toBeCloseTo(3, 5); // A[1,1] = 1 + 2*1*1 = 3
    expect(result[5]).toBeCloseTo(0, 5); // A[2,1] = 0 (unchanged)
    expect(result[6]).toBeCloseTo(2, 5); // A[0,2] = 0 + 2*1*1 = 2
    expect(result[7]).toBeCloseTo(2, 5); // A[1,2] = 0 + 2*1*1 = 2
    expect(result[8]).toBeCloseTo(3, 5); // A[2,2] = 1 + 2*1*1 = 3
  });

  test('should handle different strides', async ({ page }) => {
    const result = await page.evaluate(async () => {
      // Test with incx = 2 (every other element)
      // A = [[1, 0], [0, 1]] (identity), x = [2, 0, 3, 0] (stride 2: [2, 3])
      // x * x^T = [[2*2, 2*3], [3*2, 3*3]] = [[4, 6], [6, 9]]
      // Expected upper: A = [[1, 0], [_, 1]] + 1 * [[4, 6], [_, 9]] = [[5, 6], [_, 10]]
      const A = new Float32Array([1, 0, 0, 1]); // identity matrix, column-major
      const x = new Float32Array([2, 0, 3, 0]);
      
      await window.WebGPUBLAS.ssyr({
        uplo: 'U',
        n: 2,
        alpha: 1.0,
        x: x,
        incx: 2,
        A: A,
        lda: 2
      });
      
      return Array.from(A);
    });
    
    // Expected: [5, 0, 6, 10] (upper part updated)
    expect(result[0]).toBeCloseTo(5, 5);  // A[0,0] = 1 + 2*2 = 5
    expect(result[1]).toBeCloseTo(0, 5);  // A[1,0] = 0 (unchanged)
    expect(result[2]).toBeCloseTo(6, 5);  // A[0,1] = 0 + 2*3 = 6
    expect(result[3]).toBeCloseTo(10, 5); // A[1,1] = 1 + 3*3 = 10
  });

  test('should handle fractional values', async ({ page }) => {
    const result = await page.evaluate(async () => {
      // Test with fractional alpha and values
      // A = [[0.5, 0.25], [0.25, 1.0]], x = [0.5, 0.25], alpha = 0.5, lower triangular
      // x * x^T = [[0.5*0.5, 0.5*0.25], [0.25*0.5, 0.25*0.25]] = [[0.25, 0.125], [0.125, 0.0625]]
      // alpha * x * x^T = 0.5 * [[0.25, 0.125], [0.125, 0.0625]] = [[0.125, 0.0625], [0.0625, 0.03125]]
      // Expected lower: A = [[0.5, _], [0.25, 1.0]] + [[0.125, _], [0.0625, 0.03125]] = [[0.625, _], [0.3125, 1.03125]]
      const A = new Float32Array([0.5, 0.25, 0.25, 1.0]); // column-major
      const x = new Float32Array([0.5, 0.25]);
      
      await window.WebGPUBLAS.ssyr({
        uplo: 'L',
        n: 2,
        alpha: 0.5,
        x: x,
        incx: 1,
        A: A,
        lda: 2
      });
      
      return Array.from(A);
    });
    
    expect(result[0]).toBeCloseTo(0.625, 5);   // A[0,0] = 0.5 + 0.5*0.5*0.5 = 0.625
    expect(result[1]).toBeCloseTo(0.3125, 5); // A[1,0] = 0.25 + 0.5*0.25*0.5 = 0.3125
    expect(result[2]).toBeCloseTo(0.25, 5);   // A[0,1] = 0.25 (unchanged)
    expect(result[3]).toBeCloseTo(1.03125, 5); // A[1,1] = 1.0 + 0.5*0.25*0.25 = 1.03125
  });

  test('should handle single element matrix', async ({ page }) => {
    const result = await page.evaluate(async () => {
      // 1x1 matrix A = [2], x = [3], alpha = 0.5
      // Expected: A = [2] + 0.5 * [3*3] = [2 + 4.5] = [6.5]
      const A = new Float32Array([2]);
      const x = new Float32Array([3]);
      
      await window.WebGPUBLAS.ssyr({
        uplo: 'U',
        n: 1,
        alpha: 0.5,
        x: x,
        incx: 1,
        A: A,
        lda: 1
      });
      
      return Array.from(A);
    });
    
    expect(result[0]).toBeCloseTo(6.5, 5); // A[0,0] = 2 + 0.5*3*3 = 6.5
  });
});
