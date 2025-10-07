/**
 * Global type definitions for browser E2E tests
 */

import * as WebGPUBLAS from '../../index';

declare global {
  interface Window {
    WebGPUBLAS: typeof WebGPUBLAS;
    testRunner: {
      expect: (actual: any) => {
        toBeCloseTo: (expected: number, precision?: number) => void;
        toBe: (expected: any) => void;
      };
    };
    expect: (actual: any) => {
      toBeCloseTo: (expected: number, precision?: number) => void;
      toBe: (expected: any) => void;
    };
  }
}
