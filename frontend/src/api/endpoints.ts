import { api, API_PREFIX } from "@/api/client";
import type {
  ChatResponse,
  DashboardStats,
  DocumentDetail,
  DocumentListResponse,
  DocumentSummary,
  Entities,
  SearchResponse,
  SystemStatus,
  User,
} from "@/types";

export interface LoginPayload {
  email: string;
  password: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export const authApi = {
  login: (payload: LoginPayload) => api.post<TokenResponse>("/auth/login", payload).then((r) => r.data),
  register: (payload: LoginPayload & { full_name?: string }) =>
    api.post<TokenResponse>("/auth/register", payload).then((r) => r.data),
  me: () => api.get<User>("/auth/me").then((r) => r.data),
};

export interface DocumentQuery {
  page?: number;
  limit?: number;
  status?: string;
  doc_type?: string;
  q?: string;
  date_from?: string;
  date_to?: string;
  sort?: string;
}

export const documentsApi = {
  list: (params: DocumentQuery = {}) =>
    api.get<DocumentListResponse>("/documents", { params }).then((r) => r.data),

  detail: (id: string) => api.get<DocumentDetail>(`/documents/${id}`).then((r) => r.data),

  upload: (files: File[], onProgress?: (percent: number) => void) => {
    const form = new FormData();
    files.forEach((file) => form.append("files", file));
    return api
      .post<{ documents: DocumentSummary[]; task_id: string }>("/documents/upload", form, {
        headers: { "Content-Type": "multipart/form-data" },
        onUploadProgress: (event) => {
          if (!onProgress || !event.total) return;
          onProgress(Math.round((event.loaded / event.total) * 100));
        },
      })
      .then((r) => r.data);
  },

  verify: (
    id: string,
    payload: {
      document_type?: string;
      summary?: string;
      entities?: Entities;
      keywords?: string[];
      pages?: { page_number: number; corrected_text: string }[];
      is_verified?: boolean;
    },
  ) => api.put<DocumentDetail>(`/documents/${id}/verify`, payload).then((r) => r.data),

  reprocess: (id: string) => api.post(`/documents/${id}/reprocess`).then((r) => r.data),
  remove: (id: string) => api.delete(`/documents/${id}`),
  chat: (id: string, question: string, history: { role: "user" | "assistant"; content: string }[] = []) =>
    api.post<ChatResponse>(`/documents/${id}/chat`, { question, history }).then((r) => r.data),

  exportUrl: (id: string, format: string) => `${API_PREFIX}/documents/${id}/export?format=${format}`,
};

export const searchApi = {
  search: (payload: {
    query: string;
    mode: "hybrid" | "fulltext" | "semantic";
    filters?: Record<string, unknown>;
    limit?: number;
  }) => api.post<SearchResponse>("/search", payload).then((r) => r.data),
  exportBatchUrl: (query?: string, docType?: string) => {
    const params = new URLSearchParams({ format: "xlsx" });
    if (query) params.set("query", query);
    if (docType) params.set("doc_type", docType);
    return `${API_PREFIX}/documents/export?${params.toString()}`;
  },
};

export const systemApi = {
  status: () => api.get<SystemStatus>("/system/status").then((r) => r.data),
  stats: () => api.get<DashboardStats>("/dashboard/stats").then((r) => r.data),
  health: () => api.get("/health").then((r) => r.data),
};

export async function downloadFile(url: string, filename?: string) {
  const token = localStorage.getItem("idp-token");
  const response = await fetch(url, {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
  if (!response.ok) {
    throw new Error(`Не удалось скачать файл (${response.status})`);
  }
  const disposition = response.headers.get("Content-Disposition") ?? "";
  const match = disposition.match(/filename\*?=(?:UTF-8'')?"?([^";]+)"?/i);
  const name = filename ?? (match ? decodeURIComponent(match[1]) : "document");
  const blob = await response.blob();
  const objectUrl = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = objectUrl;
  link.download = name;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(objectUrl);
}
