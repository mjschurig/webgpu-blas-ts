import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright configuration for WebGPU e2e testing
 */
export default defineConfig({
  testDir: './src/test/e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: 'html',
  
  use: {
    baseURL: 'http://localhost:3000',
    trace: 'on-first-retry',
  },

  projects: [
    {
      name: 'chromium-webgpu',
      use: { 
        ...devices['Desktop Chrome'],
        // Enable WebGPU experimental features
        launchOptions: {
          args: [
            '--enable-unsafe-webgpu',
            '--enable-features=Vulkan,UseSkiaRenderer,WebGPU',
            '--disable-vulkan-fallback-to-gl-for-testing',
            '--use-vulkan=native',
            '--use-angle=vulkan',
            '--no-sandbox',
            '--disable-dev-shm-usage'
          ]
        }
      },
    },
  ],

  webServer: {
    command: 'npm run test:serve',
    url: 'http://localhost:3000',
    reuseExistingServer: !process.env.CI,
    timeout: 120 * 1000,
  },
});
