import { fileURLToPath, URL } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Бэкенд проксируется через dev-сервер: браузер обращается только к относительным
// URL (/api, /ws), поэтому приложение корректно работает за reverse-proxy и в preview.
// Адрес бэкенда можно переопределить переменной окружения VITE_BACKEND_URL при сборке.
const backend = process.env.VITE_BACKEND_URL ?? "http://127.0.0.1:8000";
// В облачном preview HMR идёт через HTTPS-прокси (wss:443). Локально — обычный ws,
// поэтому переключаем поведение переменной окружения VITE_HMR_PROXY=1.
const hmrProxy = process.env.VITE_HMR_PROXY === "1";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
  },
  server: {
    host: "0.0.0.0",
    port: 5173,
    strictPort: true,
    allowedHosts: true,
    hmr: hmrProxy ? { protocol: "wss", clientPort: 443 } : undefined,
    proxy: {
      "/api": { target: backend, changeOrigin: true, ws: true },
      "/ws": { target: backend, ws: true, changeOrigin: true },
    },
  },
  build: { outDir: "dist", sourcemap: false },
});
