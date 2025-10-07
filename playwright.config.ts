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
    baseURL: 'http://localhost:3001',
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
            '--use-angle=vulkan'
          ]
        }
      },
    },
    
    {
      name: 'firefox-webgpu',
      use: { 
        ...devices['Desktop Firefox'],
        // Enable WebGPU in Firefox
        launchOptions: {
          firefoxUserPrefs: {
            'dom.webgpu.enabled': true,
            'gfx.webgpu.force-enabled': true
          }
        }
      },
    },
  ],

  webServer: {
    command: 'npm run test:serve',
    url: 'http://localhost:3001',
    reuseExistingServer: !process.env.CI,
    timeout: 120 * 1000,
  },
});
