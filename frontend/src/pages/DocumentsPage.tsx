import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Download, FileText, Filter, RefreshCw, Search } from "lucide-react";
import { documentsApi, searchApi } from "@/api/endpoints";
import { downloadFile } from "@/api/endpoints";
import { toast } from "@/store/toast";
import {
  Badge,
  Card,
  EmptyState,
  Progress,
  Skeleton,
  StatusBadge,
} from "@/components/ui";
import { DOC_TYPES, type DocumentStatus } from "@/types";
import { formatBytes, formatDate, isProcessing, pluralize } from "@/lib/utils";

const STATUSES: { value: string; label: string }[] = [
  { value: "", label: "Все статусы" },
  { value: "completed", label: "Обработан" },
  { value: "verified", label: "Проверен" },
  { value: "ocr_processing", label: "Распознавание" },
  { value: "llm_processing", label: "Анализ ИИ" },
  { value: "failed", label: "Ошибка" },
];

const PAGE_SIZE = 12;

export function DocumentsPage() {
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState("");
  const [docType, setDocType] = useState("");
  const [page, setPage] = useState(1);
  const [exporting, setExporting] = useState(false);

  const params = useMemo(
    () => ({
      page,
      limit: PAGE_SIZE,
      status: status || undefined,
      doc_type: docType || undefined,
      q: query.trim() || undefined,
    }),
    [page, status, docType, query],
  );

  const documentsQuery = useQuery({
    queryKey: ["documents", params],
    queryFn: () => documentsApi.list(params),
    refetchInterval: (data) =>
      data.state.data?.items.some((item) => isProcessing(item.status)) ? 2500 : false,
  });

  const total = documentsQuery.data?.total ?? 0;
  const pages = Math.max(Math.ceil(total / PAGE_SIZE), 1);
  const items = documentsQuery.data?.items ?? [];

  const exportAll = async () => {
    setExporting(true);
    try {
      await downloadFile(searchApi.exportBatchUrl(query.trim() || undefined, docType || undefined));
      toast.success("Сводная таблица сформирована");
    } catch (error) {
      toast.error("Не удалось выполнить экспорт", String(error));
    } finally {
      setExporting(false);
    }
  };

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-white">Документы</h1>
          <p className="mt-1 text-sm text-slate-400">
            {total} {pluralize(total, ["документ", "документа", "документов"])} в локальном хранилище
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button className="btn-ghost btn-sm" onClick={() => documentsQuery.refetch()}>
            <RefreshCw className="h-3.5 w-3.5" /> Обновить
          </button>
          <button className="btn-ghost btn-sm" onClick={exportAll} disabled={exporting}>
            <Download className="h-3.5 w-3.5" /> Сводный XLSX
          </button>
          <Link to="/upload" className="btn-primary btn-sm">
            Загрузить
          </Link>
        </div>
      </div>

      <Card className="p-3">
        <div className="flex flex-wrap items-center gap-2">
          <div className="relative min-w-[220px] flex-1">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
            <input
              className="input pl-9"
              placeholder="Поиск по имени файла или тексту документа…"
              value={query}
              onChange={(event) => {
                setQuery(event.target.value);
                setPage(1);
              }}
            />
          </div>
          <div className="flex items-center gap-2">
            <Filter className="h-3.5 w-3.5 text-slate-500" />
            <select
              className="input !w-auto !py-2 text-xs"
              value={status}
              onChange={(event) => {
                setStatus(event.target.value);
                setPage(1);
              }}
            >
              {STATUSES.map((item) => (
                <option key={item.value} value={item.value}>
                  {item.label}
                </option>
              ))}
            </select>
            <select
              className="input !w-auto !py-2 text-xs"
              value={docType}
              onChange={(event) => {
                setDocType(event.target.value);
                setPage(1);
              }}
            >
              <option value="">Все типы</option>
              {DOC_TYPES.map((type) => (
                <option key={type} value={type}>
                  {type}
                </option>
              ))}
            </select>
          </div>
        </div>
      </Card>

      <Card className="overflow-hidden">
        {documentsQuery.isLoading && (
          <div className="space-y-2 p-4">
            {Array.from({ length: 5 }).map((_, index) => (
              <Skeleton key={index} className="h-16" />
            ))}
          </div>
        )}

        {!documentsQuery.isLoading && items.length === 0 && (
          <EmptyState
            icon={<FileText className="h-5 w-5" />}
            title="Документы не найдены"
            description="Измените фильтры или загрузите новые файлы."
            action={
              <Link to="/upload" className="btn-primary btn-sm mt-2">
                Загрузить документы
              </Link>
            }
          />
        )}

        {items.map((document) => (
          <div
            key={document.id}
            className="flex flex-wrap items-center gap-3 border-b border-white/[0.04] px-4 py-3.5 transition last:border-0 hover:bg-white/[0.025]"
          >
            <div className="grid h-10 w-10 shrink-0 place-items-center rounded-lg bg-white/[0.04] text-[10px] font-semibold uppercase text-slate-300">
              {(document.filename.split(".").pop() ?? "").slice(0, 4)}
            </div>
            <div className="min-w-[180px] flex-1">
              <Link
                to={`/documents/${document.id}`}
                className="block truncate text-sm font-medium text-slate-100 hover:text-brand-200"
              >
                {document.filename}
              </Link>
              <div className="mt-0.5 flex flex-wrap items-center gap-x-2 text-[11px] text-slate-500">
                <span>{formatDate(document.created_at)}</span>
                <span>·</span>
                <span>{formatBytes(document.file_size)}</span>
                <span>·</span>
                <span>
                  {document.page_count} {pluralize(document.page_count, ["стр.", "стр.", "стр."])}
                </span>
                {document.error_message && (
                  <span className="truncate text-rose-400" title={document.error_message}>
                    · {document.error_message.slice(0, 60)}
                  </span>
                )}
              </div>
              {isProcessing(document.status) && (
                <div className="mt-1.5 flex items-center gap-2">
                  <Progress value={document.progress} className="max-w-[240px]" />
                  <span className="text-[10px] text-slate-500">{document.progress}%</span>
                </div>
              )}
            </div>

            <div className="flex items-center gap-2">
              {document.doc_type && <Badge tone="muted">{document.doc_type}</Badge>}
              <StatusBadge status={document.status as DocumentStatus} />
              <button
                className="rounded-lg border border-white/10 p-1.5 text-slate-400 transition hover:border-brand-500/40 hover:text-brand-200"
                title="Экспорт JSON"
                onClick={async () => {
                  try {
                    await downloadFile(documentsApi.exportUrl(document.id, "json"));
                  } catch (error) {
                    toast.error("Экспорт не удался", String(error));
                  }
                }}
              >
                <Download className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>
        ))}
      </Card>

      {pages > 1 && (
        <div className="flex items-center justify-center gap-2">
          <button className="btn-ghost btn-sm" disabled={page <= 1} onClick={() => setPage((value) => value - 1)}>
            Назад
          </button>
          <span className="text-xs text-slate-400">
            {page} / {pages}
          </span>
          <button
            className="btn-ghost btn-sm"
            disabled={page >= pages}
            onClick={() => setPage((value) => value + 1)}
          >
            Вперёд
          </button>
        </div>
      )}
    </div>
  );
}
