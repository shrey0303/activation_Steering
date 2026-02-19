import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        timeout: 1800000,  // 30 minutes — covers long model downloads
      },
      '/health': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/api/v1/ws': {
        target: 'ws://localhost:8000',
        ws: true,
        timeout: 1800000,
      },
    },
  },
})
