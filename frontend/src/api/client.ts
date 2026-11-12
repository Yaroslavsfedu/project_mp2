import axios from "axios";

export const API_PREFIX = "/api/v1";

export const api = axios.create({
  baseURL: API_PREFIX,
  timeout: 180_000,
});

api.interceptors.request.use((config) => {
  const token = localStorage.getItem("idp-token");
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
      if (!isAuthCall) {
        localStorage.removeItem("idp-token");
        if (!window.location.pathname.startsWith("/login")) {
          window.location.href = "/login";
        }
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
