import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,       // bind to 0.0.0.0 so phone can reach it
    port: 3000,
    proxy: {
      '/auth': 'http://localhost:8000',
      '/chat': 'http://localhost:8000',
      '/documents': 'http://localhost:8000',
      '/images': 'http://localhost:8000',
      '/admin': 'http://localhost:8000',
      '/inference': 'http://localhost:8000',
      '/benchmark': 'http://localhost:8000',
      '/doctor': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
    },
  },
})
