/**
 * End-to-end tests for SGER (General Rank-1 Update)
 * 
 * Tests the WebGPU implementation against known results and reference implementations.
 */

import { test, expect } from '@playwright/test';

declare global {
  interface Window {
    WebGPUBLAS: any;
  }
}

test.describe('SGER', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('http://localhost:3000/test.html');
    await page.waitForFunction(() => window.WebGPUBLAS?.sger);
    // Initialize WebGPU context
    await page.evaluate(async () => {
      await window.WebGPUBLAS.getWebGPUContext();
    });
  });

  test('should compute basic rank-1 update', async ({ page }) => {
    const result = await page.evaluate(async () => {
      // 2x2 matrix A, x = [1, 2], y = [3, 4], alpha = 1.0
      // A = [[1, 5], [2, 6]] (column-major: [1, 2, 5, 6])
      // Expected: A = A + alpha * x * y^T = [[1, 5], [2, 6]] + 1 * [[1*3, 1*4], [2*3, 2*4]]
      //          = [[1, 5], [2, 6]] + [[3, 4], [6, 8]] = [[4, 9], [8, 14]]
      const A = new Float32Array([1, 2, 5, 6]); // column-major
      const x = new Float32Array([1, 2]);
      const y = new Float32Array([3, 4]);
      
      await window.WebGPUBLAS.sger({
        m: 2,
        n: 2,
        alpha: 1.0,
        x: x,
        incx: 1,
        y: y,
        incy: 1,
        A: A,
        lda: 2
      });
      
      return Array.from(A);
    });
    
    // Expected: [4, 8, 9, 14] (column-major: [[4, 9], [8, 14]])
    expect(result[0]).toBeCloseTo(4, 5); // A[0,0]
    expect(result[1]).toBeCloseTo(8, 5); // A[1,0]
    expect(result[2]).toBeCloseTo(9, 5); // A[0,1]
    expect(result[3]).toBeCloseTo(14, 5); // A[1,1]
  });

  test('should handle zero alpha (no operation)', async ({ page }) => {
    const result = await page.evaluate(async () => {
      // Test alpha = 0, matrix should remain unchanged
      const A = new Float32Array([1, 2, 3, 4]);
      const x = new Float32Array([5, 6]);
      const y = new Float32Array([7, 8]);
      
      await window.WebGPUBLAS.sger({
        m: 2,
        n: 2,
        alpha: 0.0,
        x: x,
        incx: 1,
        y: y,
        incy: 1,
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
      // A = [[1, 0], [0, 1]] (identity matrix), x = [1, 1], y = [1, 1]
      // Expected: A = [[1, 0], [0, 1]] + (-1) * [[1*1, 1*1], [1*1, 1*1]]
      //          = [[1, 0], [0, 1]] + [[-1, -1], [-1, -1]] = [[0, -1], [-1, 0]]
      const A = new Float32Array([1, 0, 0, 1]); // identity matrix, column-major
      const x = new Float32Array([1, 1]);
      const y = new Float32Array([1, 1]);
      
      await window.WebGPUBLAS.sger({
        m: 2,
        n: 2,
        alpha: -1.0,
        x: x,
        incx: 1,
        y: y,
        incy: 1,
        A: A,
        lda: 2
      });
      
      return Array.from(A);
    });
    
    // Expected: [0, -1, -1, 0] (column-major)
    expect(result[0]).toBeCloseTo(0, 5);  // A[0,0]
    expect(result[1]).toBeCloseTo(-1, 5); // A[1,0]
    expect(result[2]).toBeCloseTo(-1, 5); // A[0,1]
    expect(result[3]).toBeCloseTo(0, 5);  // A[1,1]
  });

  test('should handle different matrix dimensions', async ({ page }) => {
    const result = await page.evaluate(async () => {
      // 3x2 matrix A, x = [1, 2, 3], y = [1, 1], alpha = 2.0
      // A = [[1, 4], [2, 5], [3, 6]] (column-major: [1, 2, 3, 4, 5, 6])
      // x * y^T = [[1*1, 1*1], [2*1, 2*1], [3*1, 3*1]] = [[1, 1], [2, 2], [3, 3]]
      // Expected: A = [[1, 4], [2, 5], [3, 6]] + 2 * [[1, 1], [2, 2], [3, 3]]
      //          = [[1, 4], [2, 5], [3, 6]] + [[2, 2], [4, 4], [6, 6]] = [[3, 6], [6, 9], [9, 12]]
      const A = new Float32Array([1, 2, 3, 4, 5, 6]); // column-major
      const x = new Float32Array([1, 2, 3]);
      const y = new Float32Array([1, 1]);
      
      await window.WebGPUBLAS.sger({
        m: 3,
        n: 2,
        alpha: 2.0,
        x: x,
        incx: 1,
        y: y,
        incy: 1,
        A: A,
        lda: 3
      });
      
      return Array.from(A);
    });
    
    // Expected: [3, 6, 9, 6, 9, 12] (column-major)
    expect(result[0]).toBeCloseTo(3, 5);  // A[0,0]
    expect(result[1]).toBeCloseTo(6, 5);  // A[1,0]
    expect(result[2]).toBeCloseTo(9, 5);  // A[2,0]
    expect(result[3]).toBeCloseTo(6, 5);  // A[0,1]
    expect(result[4]).toBeCloseTo(9, 5);  // A[1,1]
    expect(result[5]).toBeCloseTo(12, 5); // A[2,1]
  });

  test('should handle different strides', async ({ page }) => {
    const result = await page.evaluate(async () => {
      // Test with incx = 2, incy = 2 (every other element)
      // A = [[1, 0], [0, 1]] (identity), x = [2, 0, 3, 0] (stride 2: [2, 3]), y = [1, 0, 2, 0] (stride 2: [1, 2])
      // x * y^T = [[2*1, 2*2], [3*1, 3*2]] = [[2, 4], [3, 6]]
      // Expected: A = [[1, 0], [0, 1]] + 1 * [[2, 4], [3, 6]] = [[3, 4], [3, 7]]
      const A = new Float32Array([1, 0, 0, 1]); // identity matrix, column-major
      const x = new Float32Array([2, 0, 3, 0]);
      const y = new Float32Array([1, 0, 2, 0]);
      
      await window.WebGPUBLAS.sger({
        m: 2,
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
    
    // Expected: [3, 3, 4, 7] (column-major)
    expect(result[0]).toBeCloseTo(3, 5); // A[0,0]
    expect(result[1]).toBeCloseTo(3, 5); // A[1,0]
    expect(result[2]).toBeCloseTo(4, 5); // A[0,1]
    expect(result[3]).toBeCloseTo(7, 5); // A[1,1]
  });

  test('should handle fractional values', async ({ page }) => {
    const result = await page.evaluate(async () => {
      // Test with fractional alpha and values
      // A = [[0.5, 0.25], [0.75, 1.0]], x = [0.5, 0.25], y = [2.0, 4.0], alpha = 0.5
      const A = new Float32Array([0.5, 0.75, 0.25, 1.0]); // column-major
      const x = new Float32Array([0.5, 0.25]);
      const y = new Float32Array([2.0, 4.0]);
      
      await window.WebGPUBLAS.sger({
        m: 2,
        n: 2,
        alpha: 0.5,
        x: x,
        incx: 1,
        y: y,
        incy: 1,
        A: A,
        lda: 2
      });
      
      return Array.from(A);
    });
    
    // x * y^T = [[0.5*2.0, 0.5*4.0], [0.25*2.0, 0.25*4.0]] = [[1.0, 2.0], [0.5, 1.0]]
    // alpha * x * y^T = 0.5 * [[1.0, 2.0], [0.5, 1.0]] = [[0.5, 1.0], [0.25, 0.5]]
    // A + alpha * x * y^T = [[0.5, 0.25], [0.75, 1.0]] + [[0.5, 1.0], [0.25, 0.5]] = [[1.0, 1.25], [1.0, 1.5]]
    expect(result[0]).toBeCloseTo(1.0, 5);  // A[0,0]
    expect(result[1]).toBeCloseTo(1.0, 5);  // A[1,0]
    expect(result[2]).toBeCloseTo(1.25, 5); // A[0,1]
    expect(result[3]).toBeCloseTo(1.5, 5);  // A[1,1]
  });

  test('should handle larger matrices', async ({ page }) => {
    const result = await page.evaluate(async () => {
      // 4x3 matrix, all zeros initially
      const A = new Float32Array(12); // 4x3 matrix, all zeros
      const x = new Float32Array([1, 1, 1, 1]); // all ones
      const y = new Float32Array([1, 2, 3]); // [1, 2, 3]
      
      await window.WebGPUBLAS.sger({
        m: 4,
        n: 3,
        alpha: 1.0,
        x: x,
        incx: 1,
        y: y,
        incy: 1,
        A: A,
        lda: 4
      });
      
      return Array.from(A);
    });
    
    // x * y^T = [[1*1, 1*2, 1*3], [1*1, 1*2, 1*3], [1*1, 1*2, 1*3], [1*1, 1*2, 1*3]]
    //         = [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]
    // Column-major: [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    expect(result[0]).toBeCloseTo(1, 5);  // A[0,0]
    expect(result[1]).toBeCloseTo(1, 5);  // A[1,0]
    expect(result[2]).toBeCloseTo(1, 5);  // A[2,0]
    expect(result[3]).toBeCloseTo(1, 5);  // A[3,0]
    expect(result[4]).toBeCloseTo(2, 5);  // A[0,1]
    expect(result[5]).toBeCloseTo(2, 5);  // A[1,1]
    expect(result[6]).toBeCloseTo(2, 5);  // A[2,1]
    expect(result[7]).toBeCloseTo(2, 5);  // A[3,1]
    expect(result[8]).toBeCloseTo(3, 5);  // A[0,2]
    expect(result[9]).toBeCloseTo(3, 5);  // A[1,2]
    expect(result[10]).toBeCloseTo(3, 5); // A[2,2]
    expect(result[11]).toBeCloseTo(3, 5); // A[3,2]
  });
});
