import { defineConfig, devices } from '@playwright/test';

const apiPort = process.env.PLAYWRIGHT_API_PORT ?? '8000';
const webPort = process.env.PLAYWRIGHT_WEB_PORT ?? '4173';
const apiPython = process.env.PLAYWRIGHT_PYTHON
  ?? (process.platform === 'win32'
    ? '../../.venv/Scripts/python.exe'
    : '../../.venv/bin/python');

export default defineConfig({
  testDir: './e2e',
  fullyParallel: false,
  retries: process.env.CI ? 2 : 0,
  reporter: [['list'], ['html', { open: 'never' }]],
  use: {
    baseURL: `http://127.0.0.1:${webPort}`,
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
  },
  // Seed the deterministic demo corpus with `python scripts/beatforge.py seed` first.
  webServer: [
    {
      command: `"${apiPython}" -m uvicorn beatforge_api.main:app --app-dir ../api --port ${apiPort}`,
      url: `http://127.0.0.1:${apiPort}/api/health`,
      // The desktop workflow intentionally keeps the local API available while
      // Playwright runs. Reusing a healthy server also avoids a port collision
      // when the task runner sets CI=true; Playwright still starts it when absent.
      reuseExistingServer: true,
      timeout: 120_000,
    },
    {
      command: `pnpm exec vite --host 127.0.0.1 --port ${webPort}`,
      url: `http://127.0.0.1:${webPort}`,
      reuseExistingServer: !process.env.CI,
      timeout: 120_000,
    },
  ],
  projects: [{ name: 'chrome', use: { ...devices['Desktop Chrome'], channel: 'chrome' } }],
});
