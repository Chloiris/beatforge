import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.VITE_API_PROXY
          ?? `http://127.0.0.1:${process.env.PLAYWRIGHT_API_PORT ?? '8000'}`,
        changeOrigin: true,
      },
    },
  },
  preview: { port: 4173 },
  test: {
    environment: 'jsdom',
    setupFiles: './tests/setup.ts',
    include: ['tests/**/*.test.{ts,tsx}'],
    css: true,
    coverage: { reporter: ['text', 'html'] },
  },
});
