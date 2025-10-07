# WebGPU BLAS TypeScript

A high-performance **WebGPU-accelerated BLAS** (Basic Linear Algebra Subprograms) implementation in **TypeScript**. This library provides GPU-accelerated linear algebra operations that run directly in modern web browsers and Node.js environments with WebGPU support.

## 🚀 Features

- **🔥 GPU-Accelerated**: Real WebGPU compute shaders for maximum performance
- **📊 BLAS Level 1**: Essential vector operations (AXPY, ASUM, etc.)
- **🎯 Type-Safe**: Full TypeScript support with comprehensive type definitions
- **🌐 Cross-Platform**: Works in browsers and Node.js with WebGPU
- **⚡ High Performance**: Optimized for large datasets and parallel processing
- **🧪 Thoroughly Tested**: End-to-end browser tests with actual WebGPU execution

## 📦 Installation

```bash
npm install webgpu-blas-ts
```

## 🔧 Requirements

- **WebGPU-compatible browser** (Chrome/Edge 113+, Firefox with flag enabled)
- **Modern GPU** with Vulkan/D3D12/Metal support
- **Node.js 18+** (for server-side WebGPU)

## 🏁 Quick Start

```typescript
import { initialize, sasum, saxpy, isWebGPUSupported } from 'webgpu-blas-ts';

// Check WebGPU support
if (!isWebGPUSupported()) {
  console.error('WebGPU not supported in this environment');
  process.exit(1);
}

// Initialize the library
await initialize();

// SASUM: Sum of absolute values
const x = new Float32Array([1, -2, 3, -4, 5]);
const result = await sasum({ n: 5, sx: x });
console.log(result); // 15.0

// SAXPY: y = alpha * x + y
const alpha = 2.0;
const sx = new Float32Array([1, 2, 3, 4]);
const sy = new Float32Array([5, 6, 7, 8]);
await saxpy({ n: 4, alpha, sx, sy });
console.log(Array.from(sy)); // [7, 10, 13, 16]
```

## 📖 API Reference

### Initialization

```typescript
// Check if WebGPU is supported
function isWebGPUSupported(): boolean

// Initialize WebGPU context (call once before using BLAS functions)
async function initialize(): Promise<void>
```

### BLAS Level 1 Operations

*Note: This library focuses on single precision due to WebGPU limitations with double precision.*

#### SASUM - Sum of Absolute Values
```typescript
async function sasum(params: {
  n: number;           // Number of elements
  sx: Float32Array;    // Input vector
  incx?: number;       // Stride (default: 1)
}): Promise<number>
```

#### SAXPY - Alpha X Plus Y
```typescript
// Single precision: y = alpha * x + y
async function saxpy(params: {
  n: number;           // Number of elements
  alpha: number;       // Scalar multiplier
  sx: Float32Array;    // Input vector x
  sy: Float32Array;    // Input/output vector y
  incx?: number;       // Stride for x (default: 1)
  incy?: number;       // Stride for y (default: 1)
}): Promise<void>
```

## 🧪 Testing

This library uses **end-to-end testing** with real WebGPU execution in browsers to ensure shader correctness:

```bash
# Install browser dependencies
npm run test:install-browsers

# Run e2e tests (requires WebGPU-compatible browser)
npm test

# Run tests with visible browser (for debugging)
npm run test:e2e:headed

# Interactive test runner
npm run test:e2e:ui
```

### Why E2E Testing?

- ✅ **Tests actual WebGPU shaders**, not CPU mocks
- ✅ **Validates GPU compute pipeline** setup and execution  
- ✅ **Catches hardware-specific issues** and driver compatibility
- ✅ **Ensures cross-browser compatibility** (Chrome, Firefox, etc.)
- ✅ **Performance validation** on real GPU hardware

## 🎯 Performance

WebGPU BLAS operations are designed for **large datasets** where GPU parallelization provides significant benefits:

```typescript
// Example: Large vector operations
const n = 1000000;
const x = new Float32Array(n).map(() => Math.random());
const y = new Float32Array(n).map(() => Math.random());

console.time('GPU SAXPY');
await saxpy({ n, alpha: 2.0, sx: x, sy: y });
console.timeEnd('GPU SAXPY'); // ~10-50ms on modern GPU vs seconds on CPU
```

### When to Use WebGPU BLAS

- ✅ **Large vectors/matrices** (>1000 elements)
- ✅ **Batch operations** on multiple datasets
- ✅ **Real-time applications** requiring low latency
- ✅ **Scientific computing** and machine learning workloads

## 🔧 Development

```bash
# Install dependencies  
npm install

# Build the library
npm run build

# Development mode (watch for changes)
npm run dev

# Run linting
npm run lint

# Format code
npm run format
```

### Project Structure

```
src/
├── blas/level1/          # BLAS Level 1 implementations
│   ├── sasum.ts          # Single precision absolute sum  
│   └── saxpy.ts          # Single precision AXPY
├── shaders/              # WebGPU compute shaders
│   ├── sasum_stage1.wgsl # SASUM reduction stage 1
│   ├── sasum_stage2.wgsl # SASUM reduction stage 2
│   └── saxpy.wgsl        # SAXPY implementation
├── webgpu/               # WebGPU context and utilities
│   └── context.ts        # WebGPU device management
├── types/                # TypeScript type definitions
│   └── wgsl.d.ts         # WGSL module declarations
└── test/e2e/             # End-to-end browser tests
    ├── webgpu-blas.spec.ts
    ├── test.html
    └── test-server.ts
```

## 🌐 Browser Compatibility

| Browser | Support | Notes |
|---------|---------|-------|
| **Chrome 113+** | ✅ Full | Best performance and stability |
| **Edge 113+** | ✅ Full | Chromium-based, same as Chrome |
| **Firefox 115+** | ⚠️ Experimental | Requires `dom.webgpu.enabled=true` |
| **Safari** | 🚧 Coming Soon | WebGPU support in development |

### Enabling WebGPU

**Chrome/Edge**: WebGPU enabled by default in recent versions
**Firefox**: Go to `about:config` → Set `dom.webgpu.enabled` to `true`

## 📈 Roadmap

- 🚧 **BLAS Level 2**: Matrix-vector operations (GEMV, etc.)
- 🚧 **BLAS Level 3**: Matrix-matrix operations (GEMM, etc.)  
- 🚧 **Complex number support** (ZAXPY, ZASUM, etc.)
- 🚧 **Sparse matrix operations**
- 🚧 **Advanced GPU memory management**

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details on:

- Code style and standards
- Testing requirements (e2e WebGPU tests)
- Shader development guidelines
- Performance benchmarking

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **WebGPU Working Group** for the WebGPU specification
- **BLAS Reference Implementation** for mathematical standards
- **GPU compute shader community** for optimization techniques

---

**⚡ Accelerate your linear algebra with the power of WebGPU! ⚡**