import axios from "axios";
import { useAuth } from "@/store/auth";

export const API_PREFIX = "/api/v1";

export const api = axios.create({
  baseURL: API_PREFIX,
  timeout: 180_000,
});

api.interceptors.request.use((config) => {
  const token = useAuth.getState().token;
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error?.response?.status === 401) {
      const isAuthCall = String(error.config?.url ?? "").includes("/auth/");
      // Сбрасываем сессию только если отклонён ТЕКУЩИЙ токен. Запоздалый ответ на запрос со старым
      // токеном не должен выбивать свежий вход. Переход на /login делает защищённый маршрут (без перезагрузки).
      const sentHeader = String(error.config?.headers?.Authorization ?? "");
      const current = useAuth.getState().token;
      if (!isAuthCall && current && sentHeader === `Bearer ${current}`) {
        useAuth.getState().logout();
      }
    }
    return Promise.reject(error);
  },
);

export function apiError(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const detail = error.response?.data?.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return detail.map((item: { msg?: string }) => item.msg ?? "Ошибка валидации").join("; ");
    }
    if (error.code === "ECONNABORTED") return "Превышено время ожидания запроса";
    return error.message;
  }
  return String(error);
}

export function fileUrl(path: string | null | undefined): string | null {
  if (!path) return null;
  if (path.startsWith("http")) return path;
  return path.startsWith("/api") ? path : `${API_PREFIX}${path}`;
}
