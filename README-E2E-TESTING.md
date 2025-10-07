# WebGPU BLAS E2E Testing Guide

This project now supports **end-to-end testing** of actual WebGPU shader implementations in real browser environments, alongside traditional unit tests.

## Test Architecture

### 🏗️ **Two-Tier Testing Strategy**

1. **Unit Tests (Jest + Node.js)**
   - Fast validation of JavaScript/TypeScript logic
   - Mock WebGPU APIs for CI/CD compatibility
   - Focus on edge cases, error handling, parameter validation

2. **E2E Tests (Playwright + Real Browser)**
   - Test actual WebGPU shader execution
   - Validate GPU compute pipeline performance
   - Real hardware acceleration testing

## 🚀 Quick Start

### Install Browser Dependencies
```bash
npm run test:install-browsers
```

### Run All Tests
```bash
# Unit tests (Node.js with mocks)
npm test

# E2E tests (Real browsers with WebGPU)
npm run test:e2e

# E2E tests with browser UI visible
npm run test:e2e:headed

# Interactive test debugging
npm run test:e2e:ui
```

## 📁 File Structure

```
src/test/
├── setup.ts                 # Jest setup (WebGPU API mocks for Node.js)
├── blas/level1/             # Unit tests (existing)
│   ├── sasum.test.ts
│   ├── saxpy.test.ts
│   └── ...
└── e2e/                     # End-to-end tests
    ├── webgpu-blas.spec.ts  # Playwright browser tests
    ├── test-server.ts       # Development server
    └── test.html            # Test runner UI
```

## 📊 What Each Test Type Validates

### Unit Tests (Jest)
- ✅ Parameter validation
- ✅ Edge case handling  
- ✅ Mathematical correctness (CPU reference)
- ✅ Error conditions
- ❌ **NOT testing actual WebGPU shaders**

### E2E Tests (Playwright)
- ✅ **Real WebGPU shader execution**
- ✅ GPU buffer management
- ✅ Compute pipeline setup
- ✅ Performance characteristics
- ✅ Cross-browser compatibility
- ✅ Hardware-specific behavior

## 🌐 Browser Compatibility

### Supported Browsers
- **Chrome/Chromium** (Primary target)
  - Requires `--enable-unsafe-webgpu` flag
  - Best WebGPU support
- **Firefox** 
  - Requires `dom.webgpu.enabled=true`
  - Experimental support
- **Edge**
  - Same as Chrome (Chromium-based)

### WebGPU Requirements
- Modern GPU with Vulkan/D3D12/Metal support
- Updated graphics drivers
- WebGPU feature flag enabled

## 🧪 Test Examples

### Basic SASUM Test
```typescript
test('should compute sum of absolute values', async ({ page }) => {
  const result = await page.evaluate(async () => {
    const sx = new Float32Array([1, -2, 3, -4, 5]);
    return await window.WebGPUBLAS.sasum({ n: 5, sx });
  });
  
  expect(result).toBeCloseTo(15.0, 4);
});
```

### Performance Test
```typescript
test('should handle large vectors efficiently', async ({ page }) => {
  const { result, duration } = await page.evaluate(async () => {
    const n = 10000;
    const sx = new Float32Array(n).fill(1);
    
    const start = performance.now();
    const result = await window.WebGPUBLAS.sasum({ n, sx });
    const end = performance.now();
    
    return { result, duration: end - start };
  });
  
  expect(result).toBeCloseTo(10000);
  expect(duration).toBeLessThan(100); // GPU should be fast!
});
```

## 🔧 Configuration

### Playwright Config (`playwright.config.ts`)
- **Chrome**: WebGPU flags enabled
- **Firefox**: WebGPU preferences set
- **Test Server**: Automatic startup on port 3001
- **Retries**: 2x in CI, 0x locally

### Jest Config (`jest.config.js`)
- **Excludes**: E2E tests from Jest runs
- **Includes**: Traditional unit tests only
- **Mocks**: WebGPU APIs for Node.js compatibility

## 🐛 Debugging

### WebGPU Not Supported
```bash
# Check browser support
npm run test:e2e:headed
# Look for error messages in browser console
```

### Shader Compilation Errors
```bash
# Run with visible browser
npm run test:e2e:headed
# Check DevTools → Console for WebGPU errors
```

### Performance Issues
```bash
# Use interactive test runner
npm run test:e2e:ui
# Monitor GPU usage in browser DevTools
```

## 🎯 Best Practices

### When to Use Unit Tests
- Parameter validation
- Edge cases (empty arrays, negative values)
- Mathematical accuracy verification
- Fast CI/CD feedback loops

### When to Use E2E Tests  
- **WebGPU shader functionality**
- Performance benchmarking
- Cross-browser compatibility
- Hardware-specific behavior
- Integration testing

### Test Data Strategy
```typescript
// Good: Test mathematical properties
const testCases = [
  { input: [1, 2, 3], expected: 6 },
  { input: [-1, -2, -3], expected: 6 },
];

// Good: Test edge cases
const edgeCases = [
  { n: 0, expected: 0 },
  { incx: 0, expected: 0 },
];

// Good: Test performance
const performanceTests = [
  { size: 1000, maxTime: 10 },
  { size: 100000, maxTime: 100 },
];
```

## 🚨 Common Pitfalls

1. **Don't mock WebGPU in E2E tests** - That defeats the purpose
2. **Don't run E2E tests in CI without GPU** - They'll fail
3. **Don't test only mathematical correctness** - Also test GPU behavior
4. **Don't ignore browser compatibility** - Test in multiple browsers
5. **Don't skip performance tests** - GPU performance is the main benefit

---

**The goal**: Ensure your WebGPU shaders actually work correctly on real hardware, not just pass mocked CPU tests! 🎯
