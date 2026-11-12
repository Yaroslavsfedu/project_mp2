export type DocumentStatus =
  | "pending"
  | "ocr_processing"
  | "llm_processing"
  | "completed"
  | "verified"
  | "failed";

export interface User {
  id: string;
  email: string;
  full_name: string;
  role: "admin" | "operator" | "viewer";
  created_at: string;
}

export interface BoundingBox {
  id: string;
  text: string;
  confidence: number;
  bbox: [number, number, number, number];
}

export interface OcrPage {
  page_number: number;
  raw_text: string;
  corrected_text: string | null;
  bounding_boxes: BoundingBox[];
  image_width: number;
  image_height: number;
  avg_confidence: number;
  source: string;
  image_url: string | null;
}

export interface DocumentSummary {
  id: string;
  filename: string;
  mime_type: string;
  file_size: number;
  page_count: number;
  status: DocumentStatus;
  progress: number;
  doc_type: string | null;
  summary: string | null;
  is_verified: boolean;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

export interface DocumentDetail extends DocumentSummary {
  pages: OcrPage[];
  entities: Entities;
  keywords: string[];
  analysis_engine: string | null;
  document_url: string | null;
}

export interface Entities {
  persons: string[];
  organizations: string[];
  dates: string[];
  amounts: string[];
  inn_ogrn: string[];
  document_number: string | null;
  addresses: string[];
}

export interface DocumentListResponse {
  items: DocumentSummary[];
  total: number;
  page: number;
  limit: number;
}

export interface SearchHit {
  document_id: string;
  filename: string;
  doc_type: string | null;
  status: DocumentStatus;
  page_number: number | null;
  snippet: string;
  score: number;
  match_type: string;
}

export interface SearchResponse {
  query: string;
  mode: string;
  took_ms: number;
  total: number;
  hits: SearchHit[];
}

export interface ChatSource {
  page_number: number;
  snippet: string;
}

export interface ChatResponse {
  answer: string;
  sources: ChatSource[];
  engine: string;
}

export interface DashboardStats {
  total_documents: number;
  by_status: Record<string, number>;
  by_type: Record<string, number>;
  verified_count: number;
  pages_processed: number;
  avg_confidence: number;
  last_7_days: { date: string; count: number }[];
}

export interface SystemStatus {
  llm: {
    provider: string;
    url: string;
    model: string;
    embedding_model: string;
    vision_model?: string;
    available: boolean;
    mode: string;
    installed_models?: string[];
    model_installed?: boolean;
    embedding_model_installed?: boolean;
    vision_model_installed?: boolean;
  };
  ocr: {
    engine: string;
    configured: string;
    languages: string;
    tesseract_installed: boolean;
    preprocess_enabled: boolean;
  };
  gpu: {
    available: boolean;
    name?: string;
    memory_mb?: number;
    driver?: string;
    recommended_llm?: string;
    recommended_vision?: string;
    reason?: string;
  };
  cpu: { cores: number; note?: string };
  storage_backend: string;
  queue_backend: string;
  max_upload_mb: number;
  hints: string[];
}

export interface AdminUser {
  id: string;
  email: string;
  full_name: string;
  role: "admin" | "operator" | "viewer";
  is_active: boolean;
  created_at: string;
  documents_count: number;
}

export interface AuditEntry {
  id: string;
  user_email: string;
  action: string;
  document_id: string | null;
  document_name: string;
  details: Record<string, unknown>;
  created_at: string;
}

export interface AdminOverview {
  users_total: number;
  documents_total: number;
  pages_total: number;
  verified_total: number;
  storage_bytes: number;
  audit_events: number;
  actions_24h: { action: string; count: number }[];
}

export interface QualityMetrics {
  cer: number;
  wer: number;
  ner_precision: number;
  ner_recall: number;
  ner_f1: number;
  doc_type_accuracy: number;
  avg_duration_sec: number;
}

export interface QualityDocumentRow {
  filename: string;
  status: string;
  error: string | null;
  pages: number;
  duration_sec: number;
  cer: number;
  wer: number;
  doc_type_expected: string;
  doc_type_predicted: string;
  doc_type_correct: boolean;
  llm_engine: string;
  ocr_source: string;
}

export interface QualityReport {
  generated_at: string;
  documents: number;
  metrics: QualityMetrics;
  per_document: QualityDocumentRow[];
  skipped: { filename: string; reason: string }[];
  environment: Record<string, unknown>;
  worst_documents?: QualityDocumentRow[];
}

export const DOC_TYPES = [
  "Договор",
  "Приказ",
  "Счет",
  "Акт",
  "Заявление",
  "Научная статья",
  "Паспорт/Удостоверение",
  "Другое",
] as const;

export const ENTITY_LABELS: Record<keyof Entities, string> = {
  persons: "ФИО",
  organizations: "Организации",
  dates: "Даты",
  amounts: "Суммы",
  inn_ogrn: "ИНН / КПП / ОГРН",
  document_number: "Номер документа",
  addresses: "Адреса",
};
