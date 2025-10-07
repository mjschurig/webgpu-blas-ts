/**
 * End-to-end tests for SGEMV (General Matrix-Vector Multiply)
 * 
 * Tests the WebGPU implementation against known results and reference implementations.
 */

import { test, expect } from '@playwright/test';

declare global {
  interface Window {
    WebGPUBLAS: any;
  }
}

test.describe('SGEMV', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('http://localhost:3000/test.html');
    await page.waitForFunction(() => window.WebGPUBLAS?.sgemv);
    // Initialize WebGPU context
    await page.evaluate(async () => {
      await window.WebGPUBLAS.getWebGPUContext();
    });
  });

  test('should compute basic matrix-vector multiply (no transpose)', async ({ page }) => {
    const result = await page.evaluate(async () => {
      // 2x3 matrix A, 3x1 vector x, 2x1 vector y
      // A = [[1, 4, 7], [2, 5, 8]], x = [1, 2, 3], y = [0, 0]
      // Expected: y = A * x = [1*1 + 4*2 + 7*3, 2*1 + 5*2 + 8*3] = [30, 36]
      const A = new Float32Array([1, 2, 4, 5, 7, 8]); // column-major
      const x = new Float32Array([1, 2, 3]);
      const y = new Float32Array([0, 0]);
      
      await window.WebGPUBLAS.sgemv({
        trans: 'N',
        m: 2,
        n: 3,
        alpha: 1.0,
        A: A,
        lda: 2,
        x: x,
        incx: 1,
        beta: 0.0,
        y: y,
        incy: 1
      });
      
      return Array.from(y);
    });
    
    expect(result[0]).toBeCloseTo(30, 5);
    expect(result[1]).toBeCloseTo(36, 5);
  });

  test('should compute basic matrix-vector multiply (transpose)', async ({ page }) => {
    const result = await page.evaluate(async () => {
      // 2x3 matrix A, 2x1 vector x, 3x1 vector y  
      // A^T = [[1, 2], [4, 5], [7, 8]], x = [1, 2], y = [0, 0, 0]
      // Expected: y = A^T * x = [1*1 + 2*2, 4*1 + 5*2, 7*1 + 8*2] = [5, 14, 23]
      const A = new Float32Array([1, 2, 4, 5, 7, 8]); // column-major
      const x = new Float32Array([1, 2]);
      const y = new Float32Array([0, 0, 0]);
      
      await window.WebGPUBLAS.sgemv({
        trans: 'T',
        m: 2,
        n: 3,
        alpha: 1.0,
        A: A,
        lda: 2,
        x: x,
        incx: 1,
        beta: 0.0,
        y: y,
        incy: 1
      });
      
      return Array.from(y);
    });
    
    expect(result[0]).toBeCloseTo(5, 5);
    expect(result[1]).toBeCloseTo(14, 5);
    expect(result[2]).toBeCloseTo(23, 5);
  });

  test('should handle alpha and beta scaling', async ({ page }) => {
    const result = await page.evaluate(async () => {
      // Test y = alpha * A * x + beta * y with alpha = 2, beta = 3
      // A = [[1, 0], [0, 1]], x = [1, 2], y = [4, 5] (initial)
      // Expected: y = 2 * [1, 2] + 3 * [4, 5] = [2, 4] + [12, 15] = [14, 19]
      const A = new Float32Array([1, 0, 0, 1]); // identity matrix, column-major
      const x = new Float32Array([1, 2]);
      const y = new Float32Array([4, 5]);
      
      await window.WebGPUBLAS.sgemv({
        trans: 'N',
        m: 2,
        n: 2,
        alpha: 2.0,
        A: A,
        lda: 2,
        x: x,
        incx: 1,
        beta: 3.0,
        y: y,
        incy: 1
      });
      
      return Array.from(y);
    });
    
    expect(result[0]).toBeCloseTo(14, 5);
    expect(result[1]).toBeCloseTo(19, 5);
  });

  test('should handle zero alpha (beta only)', async ({ page }) => {
    const result = await page.evaluate(async () => {
      // Test y = 0 * A * x + 2 * y = 2 * y
      const A = new Float32Array([1, 2, 3, 4]); // 2x2 matrix
      const x = new Float32Array([1, 1]);
      const y = new Float32Array([3, 4]);
      
      await window.WebGPUBLAS.sgemv({
        trans: 'N',
        m: 2,
        n: 2,
        alpha: 0.0,
        A: A,
        lda: 2,
        x: x,
        incx: 1,
        beta: 2.0,
        y: y,
        incy: 1
      });
      
      return Array.from(y);
    });
    
    expect(result[0]).toBeCloseTo(6, 5);
    expect(result[1]).toBeCloseTo(8, 5);
  });

  test('should handle zero beta (alpha only)', async ({ page }) => {
    const result = await page.evaluate(async () => {
      // Test y = 1 * A * x + 0 * y = A * x (ignore initial y values)
      // A = [[2, 0], [0, 3]], x = [1, 2], y = [999, 999] (initial, should be ignored)
      // Expected: y = [2, 6]
      const A = new Float32Array([2, 0, 0, 3]); // column-major
      const x = new Float32Array([1, 2]);
      const y = new Float32Array([999, 999]);
      
      await window.WebGPUBLAS.sgemv({
        trans: 'N',
        m: 2,
        n: 2,
        alpha: 1.0,
        A: A,
        lda: 2,
        x: x,
        incx: 1,
        beta: 0.0,
        y: y,
        incy: 1
      });
      
      return Array.from(y);
    });
    
    expect(result[0]).toBeCloseTo(2, 5);
    expect(result[1]).toBeCloseTo(6, 5);
  });

  test('should handle different strides', async ({ page }) => {
    const result = await page.evaluate(async () => {
      // Test with incx = 2, incy = 2 (every other element)
      // A = [[1, 2], [3, 4]], x = [1, 0, 2, 0] (stride 2: [1, 2]), y = [0, 0, 0, 0]
      // Expected at stride positions: y[0] = 1*1 + 2*2 = 5, y[2] = 3*1 + 4*2 = 11
      const A = new Float32Array([1, 3, 2, 4]); // column-major
      const x = new Float32Array([1, 0, 2, 0]);
      const y = new Float32Array([0, 0, 0, 0]);
      
      await window.WebGPUBLAS.sgemv({
        trans: 'N',
        m: 2,
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
    
    expect(result[0]).toBeCloseTo(5, 5);
    expect(result[1]).toBeCloseTo(0, 5); // unchanged
    expect(result[2]).toBeCloseTo(11, 5);
    expect(result[3]).toBeCloseTo(0, 5); // unchanged
  });

  test('should handle larger matrices', async ({ page }) => {
    const result = await page.evaluate(async () => {
      // 4x4 identity matrix, x = [1, 2, 3, 4]
      // Expected: y = x (identity transformation)
      const A = new Float32Array(16);
      for (let i = 0; i < 4; i++) {
        A[i * 4 + i] = 1; // Set diagonal elements to 1 (column-major)
      }
      
      const x = new Float32Array([1, 2, 3, 4]);
      const y = new Float32Array([0, 0, 0, 0]);
      
      await window.WebGPUBLAS.sgemv({
        trans: 'N',
        m: 4,
        n: 4,
        alpha: 1.0,
        A: A,
        lda: 4,
        x: x,
        incx: 1,
        beta: 0.0,
        y: y,
        incy: 1
      });
      
      return Array.from(y);
    });
    
    expect(result[0]).toBeCloseTo(1, 5);
    expect(result[1]).toBeCloseTo(2, 5);
    expect(result[2]).toBeCloseTo(3, 5);
    expect(result[3]).toBeCloseTo(4, 5);
  });
});
