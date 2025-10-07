import { build } from 'esbuild';
import { glsl } from 'esbuild-plugin-glsl';

const baseConfig = {
  entryPoints: ['src/index.ts'],
  bundle: true,
  target: 'es2022',
  sourcemap: true,
  plugins: [
    glsl({
      minify: true,
      resolveIncludes: true,
      preserveLegalComments: true,
    }),
  ],
  external: [
    // Mark Node.js modules as external for browser builds
    'fs',
    'path',
    'util',
  ],
};

// Build for browser (ESM)
await build({
  ...baseConfig,
  format: 'esm',
  platform: 'browser',
  outfile: 'dist/webgpu-blas.esm.js',
});

// Build for Node.js (CommonJS)
await build({
  ...baseConfig,
  platform: 'node',
  format: 'cjs',
  outfile: 'dist/webgpu-blas.cjs.js',
});

console.log('Build completed successfully!');
