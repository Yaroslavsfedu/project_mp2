import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  AlertTriangle,
  ArrowLeft,
  BadgeCheck,
  Braces,
  ChevronDown,
  Download,
  FileText,
  MessageSquare,
  RefreshCw,
  Tags,
  Trash2,
} from "lucide-react";
import { apiError } from "@/api/client";
import { documentsApi, downloadFile } from "@/api/endpoints";
import { toast } from "@/store/toast";
import { cn, formatBytes, formatDate, isProcessing } from "@/lib/utils";
import type { BoundingBox, DocumentDetail, Entities } from "@/types";
import {
  Badge,
  Card,
  EmptyState,
  ErrorBanner,
  Progress,
  Skeleton,
  StatusBadge,
} from "@/components/ui";
import { DocumentCanvas } from "@/components/viewer/DocumentCanvas";
import { EntitiesEditor } from "@/components/viewer/EntitiesEditor";
import { ChatPanel } from "@/components/chat/ChatPanel";

type Tab = "text" | "entities" | "chat" | "json";

const TABS: { id: Tab; label: string; icon: typeof FileText }[] = [
  { id: "text", label: "Текст", icon: FileText },
  { id: "entities", label: "Сущности", icon: Tags },
  { id: "chat", label: "Чат", icon: MessageSquare },
  { id: "json", label: "JSON", icon: Braces },
];

const EXPORT_FORMATS = [
  { format: "json", label: "JSON (структурированные данные)" },
  { format: "docx", label: "DOCX (карточка документа)" },
  { format: "csv", label: "CSV (сущности и текст)" },
  { format: "md", label: "Markdown" },
  { format: "txt", label: "TXT (распознанный текст)" },
];

function TextPanel({
  document,
  pageNumber,
  selectedBox,
  onSelectLine,
}: {
  document: DocumentDetail;
  pageNumber: number;
  selectedBox: BoundingBox | null;
  onSelectLine: (text: string) => void;
}) {
  const page = document.pages.find((item) => item.page_number === pageNumber) ?? document.pages[0];
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [highlightLine, setHighlightLine] = useState<number | null>(null);

  const lines = useMemo(() => {
    const text = page?.corrected_text || page?.raw_text || "";
    return text.split(/\n+/).filter((line) => line.trim().length > 0);
  }, [page]);

  useEffect(() => {
    if (!selectedBox) return;
    const normalized = selectedBox.text.trim().toLowerCase();
    const index = lines.findIndex((line) => {
      const low = line.trim().toLowerCase();
      return low === normalized || low.startsWith(normalized) || normalized.startsWith(low) || low.includes(normalized);
    });
    setHighlightLine(index >= 0 ? index : null);
    if (index >= 0) {
      const element = containerRef.current?.querySelector<HTMLElement>(`[data-line="${index}"]`);
      element?.scrollIntoView({ block: "center", behavior: "smooth" });
    }
  }, [selectedBox, lines]);

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex flex-wrap items-center gap-2 border-b border-white/[0.06] px-4 py-2 text-[11px] text-slate-400">
        <span>
          Страница {page?.page_number} из {document.pages.length}
        </span>
        <span>·</span>
        <span>{lines.length} строк</span>
        <span>·</span>
        <span>{page?.bounding_boxes.length ?? 0} рамок</span>
        <span className="ml-auto">
          уверенность {(Number(page?.avg_confidence ?? 0) * 100).toFixed(1)}%
        </span>
      </div>
      <div ref={containerRef} className="flex-1 overflow-y-auto px-4 py-3">
        <div className="space-y-0.5 font-mono text-[12.5px] leading-relaxed text-slate-300">
          {lines.map((line, index) => (
            <p
              key={index}
              data-line={index}
              onClick={() => onSelectLine(line)}
              className={cn(
                "cursor-pointer rounded px-1.5 py-0.5 transition hover:bg-white/[0.05]",
                highlightLine === index && "bg-brand-500/20 text-white ring-1 ring-inset ring-brand-500/30",
              )}
            >
              {line}
            </p>
          ))}
          {lines.length === 0 && (
            <p className="text-slate-500">
              Текст отсутствует на этой странице
              {page?.source === "needs_ocr" && " — требуется OCR-движок (Tesseract с языком rus)"}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

export function DocumentViewPage() {
  const { id = "" } = useParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [tab, setTab] = useState<Tab>("text");
  const [pageNumber, setPageNumber] = useState(1);
  const [selectedBox, setSelectedBox] = useState<BoundingBox | null>(null);
  const [exportOpen, setExportOpen] = useState(false);
  const [exporting, setExporting] = useState<string | null>(null);

  const documentQuery = useQuery({
    queryKey: ["document", id],
    queryFn: () => documentsApi.detail(id),
    enabled: Boolean(id),
    refetchInterval: (data) => {
      const status = data.state.data?.status;
      return status && isProcessing(status) ? 1500 : false;
    },
  });

  const document = documentQuery.data;

  const reprocess = useMutation({
    mutationFn: () => documentsApi.reprocess(id),
    onSuccess: () => {
      toast.info("Повторная обработка запущена");
      queryClient.invalidateQueries({ queryKey: ["document", id] });
    },
    onError: (error) => toast.error("Не удалось запустить обработку", apiError(error)),
  });

  const remove = useMutation({
    mutationFn: () => documentsApi.remove(id),
    onSuccess: () => {
      toast.success("Документ удалён");
      queryClient.invalidateQueries({ queryKey: ["documents"] });
      queryClient.invalidateQueries({ queryKey: ["stats"] });
      navigate("/documents");
    },
    onError: (error) => toast.error("Не удалось удалить документ", apiError(error)),
  });

  useEffect(() => {
    setPageNumber(1);
    setSelectedBox(null);
  }, [id]);

  const currentPage = document?.pages.find((page) => page.page_number === pageNumber) ?? document?.pages[0];

  if (documentQuery.isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-10 w-1/2" />
        <div className="grid gap-4 lg:grid-cols-2">
          <Skeleton className="h-[70vh]" />
          <Skeleton className="h-[70vh]" />
        </div>
      </div>
    );
  }

  if (documentQuery.isError || !document) {
    return (
      <Card className="p-6">
        <ErrorBanner message={apiError(documentQuery.error) || "Документ не найден"} />
        <Link to="/documents" className="btn-ghost btn-sm mt-4">
          <ArrowLeft className="h-3.5 w-3.5" /> К списку документов
        </Link>
      </Card>
    );
  }

  const processing = isProcessing(document.status);
  const entities = document.entities as Entities;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <Link
            to="/documents"
            className="mb-2 inline-flex items-center gap-1.5 text-xs text-slate-400 transition hover:text-slate-200"
          >
            <ArrowLeft className="h-3.5 w-3.5" /> Все документы
          </Link>
          <h1 className="truncate text-xl font-semibold tracking-tight text-white" title={document.filename}>
            {document.filename}
          </h1>
          <div className="mt-1.5 flex flex-wrap items-center gap-2 text-[11px] text-slate-400">
            <StatusBadge status={document.status} />
            {document.doc_type && <Badge tone="muted">{document.doc_type}</Badge>}
            {document.is_verified && (
              <Badge tone="brand">
                <BadgeCheck className="h-3 w-3" /> проверен
              </Badge>
            )}
            <span>{formatDate(document.created_at)}</span>
            <span>·</span>
            <span>{formatBytes(document.file_size)}</span>
            <span>·</span>
            <span>{document.pages.length} стр.</span>
            {document.analysis_engine && (
              <>
                <span>·</span>
                <span>движок: {document.analysis_engine}</span>
              </>
            )}
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <div className="relative">
            <button className="btn-ghost btn-sm" onClick={() => setExportOpen((value) => !value)}>
              <Download className="h-3.5 w-3.5" /> Экспорт <ChevronDown className="h-3 w-3" />
            </button>
            {exportOpen && (
              <>
                <div className="fixed inset-0 z-10" onClick={() => setExportOpen(false)} />
                <div className="absolute right-0 z-20 mt-1 w-64 overflow-hidden rounded-xl border border-white/10 bg-ink-850 shadow-xl">
                  {EXPORT_FORMATS.map((item) => (
                    <button
                      key={item.format}
                      className="flex w-full items-center justify-between px-3 py-2 text-left text-xs text-slate-300 transition hover:bg-white/[0.06] hover:text-white disabled:opacity-50"
                      disabled={exporting === item.format}
                      onClick={async () => {
                        setExporting(item.format);
                        try {
                          await downloadFile(documentsApi.exportUrl(document.id, item.format));
                          toast.success(`Экспорт ${item.format.toUpperCase()} готов`);
                        } catch (error) {
                          toast.error("Не удалось экспортировать", String(error));
                        } finally {
                          setExporting(null);
                          setExportOpen(false);
                        }
                      }}
                    >
                      {item.label}
                      <span className="text-[10px] uppercase text-slate-500">{item.format}</span>
                    </button>
                  ))}
                </div>
              </>
            )}
          </div>
          <button className="btn-ghost btn-sm" onClick={() => reprocess.mutate()} disabled={reprocess.isPending}>
            <RefreshCw className={cn("h-3.5 w-3.5", reprocess.isPending && "animate-spin")} /> Повторить
          </button>
          <button
            className="btn-danger btn-sm"
            onClick={() => {
              if (window.confirm(`Удалить «${document.filename}»? Действие необратимо.`)) remove.mutate();
            }}
            disabled={remove.isPending}
          >
            <Trash2 className="h-3.5 w-3.5" /> Удалить
          </button>
        </div>
      </div>

      {processing && (
        <Card className="p-4">
          <div className="flex items-center justify-between text-sm">
            <span className="flex items-center gap-2 text-slate-200">
              <RefreshCw className="h-4 w-4 animate-spin text-brand-300" />
              Идёт обработка документа — {document.progress}%
            </span>
            <span className="text-xs text-slate-400">статус обновится автоматически по WebSocket</span>
          </div>
          <Progress value={document.progress} className="mt-2.5" />
        </Card>
      )}

      {document.status === "failed" && (
        <Card className="border-rose-500/25 bg-rose-500/[0.06] p-4">
          <div className="flex items-start gap-2.5">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-rose-300" />
            <div>
              <p className="text-sm font-semibold text-rose-200">Обработка завершилась ошибкой</p>
              <p className="mt-1 text-xs leading-relaxed text-rose-200/80">{document.error_message}</p>
              <p className="mt-2 text-[11px] text-rose-200/60">
                Для сканов требуется OCR-движок. В Docker-образе он устанавливается автоматически
                (tesseract-ocr + tesseract-ocr-rus), в локальном режиме без OCR обрабатываются PDF с текстовым
                слоем и DOCX.
              </p>
            </div>
          </div>
        </Card>
      )}

      <div className="grid min-h-[calc(100vh-320px)] gap-4 xl:grid-cols-2">
        <Card className="flex min-h-[560px] flex-col overflow-hidden">
          {currentPage ? (
            <DocumentCanvas
              page={currentPage}
              pageCount={document.pages.length}
              pageNumber={pageNumber}
              onPageChange={(page) => {
                setPageNumber(page);
                setSelectedBox(null);
              }}
              selectedBoxId={selectedBox?.id ?? null}
              onSelectBox={(box) => {
                setSelectedBox(box);
                setTab("text");
              }}
            />
          ) : (
            <EmptyState title="Страницы недоступны" description="Документ ещё не обработан" />
          )}
        </Card>

        <Card className="flex min-h-[560px] flex-col overflow-hidden">
          <div className="flex items-center gap-1 border-b border-white/[0.06] px-2 py-1.5">
            {TABS.map((item) => (
              <button
                key={item.id}
                onClick={() => setTab(item.id)}
                className={cn(
                  "flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs font-medium transition",
                  tab === item.id
                    ? "bg-brand-600/20 text-brand-100"
                    : "text-slate-400 hover:bg-white/[0.04] hover:text-slate-200",
                )}
              >
                <item.icon className="h-3.5 w-3.5" /> {item.label}
              </button>
            ))}
            {selectedBox && (
              <span className="ml-auto mr-1 max-w-[45%] truncate rounded bg-brand-500/15 px-2 py-1 text-[10px] text-brand-100">
                {selectedBox.text}
              </span>
            )}
          </div>

          <div className="flex min-h-0 flex-1 flex-col">
            {tab === "text" && (
              <TextPanel
                document={document}
                pageNumber={pageNumber}
                selectedBox={selectedBox}
                onSelectLine={(text) => {
                  const boxes = currentPage?.bounding_boxes ?? [];
                  const normalized = text.trim().toLowerCase();
                  const match =
                    boxes.find((box) => box.text.trim().toLowerCase() === normalized) ??
                    boxes.find((box) => normalized.includes(box.text.trim().toLowerCase())) ??
                    boxes.find((box) => box.text.trim().toLowerCase().includes(normalized.slice(0, 30)));
                  if (match) setSelectedBox(match);
                  else toast.info("Для этой строки нет рамки на скане");
                }}
              />
            )}

            {tab === "entities" && (
              <div className="min-h-0 flex-1 overflow-y-auto px-4 pt-4">
                <EntitiesEditor
                  document={document}
                  onSaved={() => {
                    queryClient.invalidateQueries({ queryKey: ["document", id] });
                    queryClient.invalidateQueries({ queryKey: ["documents"] });
                    queryClient.invalidateQueries({ queryKey: ["stats"] });
                  }}
                />
              </div>
            )}

            {tab === "chat" && <ChatPanel documentId={document.id} />}

            {tab === "json" && (
              <div className="min-h-0 flex-1 overflow-auto p-4">
                <pre className="whitespace-pre-wrap break-words rounded-xl border border-white/[0.06] bg-black/30 p-3 font-mono text-[11px] leading-relaxed text-slate-300">
                  {JSON.stringify(
                    {
                      id: document.id,
                      filename: document.filename,
                      status: document.status,
                      doc_type: document.doc_type,
                      summary: document.summary,
                      keywords: document.keywords,
                      entities,
                      pages: document.pages.map((page) => ({
                        page_number: page.page_number,
                        source: page.source,
                        avg_confidence: page.avg_confidence,
                        blocks: page.bounding_boxes.length,
                      })),
                    },
                    null,
                    2,
                  )}
                </pre>
                <button
                  className="btn-ghost btn-sm mt-3"
                  onClick={async () => {
                    try {
                      await navigator.clipboard.writeText(
                        JSON.stringify({ ...document, pages: document.pages }, null, 2),
                      );
                      toast.success("JSON скопирован в буфер обмена");
                    } catch {
                      toast.error("Не удалось скопировать");
                    }
                  }}
                >
                  Скопировать JSON
                </button>
              </div>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
}
