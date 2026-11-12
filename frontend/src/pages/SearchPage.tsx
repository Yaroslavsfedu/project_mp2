import { useState } from "react";
import { Link } from "react-router-dom";
import { useMutation } from "@tanstack/react-query";
import { Download, FileSearch, Search, Sparkles, Target, Waves } from "lucide-react";
import { searchApi, downloadFile } from "@/api/endpoints";
import { apiError } from "@/api/client";
import { toast } from "@/store/toast";
import { cn, highlight, pluralize } from "@/lib/utils";
import { Badge, Card, EmptyState, ErrorBanner, Spinner, StatusBadge } from "@/components/ui";
import { DOC_TYPES } from "@/types";

type Mode = "hybrid" | "fulltext" | "semantic";

const MODES: { id: Mode; label: string; hint: string; icon: typeof Search }[] = [
  { id: "hybrid", label: "Гибрид", hint: "FTS + векторный индекс", icon: Sparkles },
  { id: "fulltext", label: "Точный", hint: "Морфология и подсветка", icon: Target },
  { id: "semantic", label: "Семантический", hint: "Поиск по смыслу", icon: Waves },
];

const EXAMPLES = [
  "неустойка за просрочку поставки",
  "сумма договора и реквизиты",
  "кто назначен руководителем лаборатории",
  "анализ документов и OCR",
];

export function SearchPage() {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<Mode>("hybrid");
  const [docType, setDocType] = useState("");
  const [status, setStatus] = useState("");
  const [exporting, setExporting] = useState(false);

  const search = useMutation({
    mutationFn: (payload: { query: string; mode: Mode; docType: string; status: string }) =>
      searchApi.search({
        query: payload.query,
        mode: payload.mode,
        filters: {
          ...(payload.docType ? { doc_type: payload.docType } : {}),
          ...(payload.status ? { status: payload.status } : {}),
        },
        limit: 25,
      }),
    onError: (error) => toast.error("Поиск не выполнен", apiError(error)),
  });

  const run = (nextQuery = query) => {
    const value = nextQuery.trim();
    if (!value) return;
    setQuery(nextQuery);
    search.mutate({ query: value, mode, docType, status });
  };

  const result = search.data;

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-white">Умный поиск</h1>
        <p className="mt-1 text-sm text-slate-400">
          Поиск по распознанному тексту всех документов: полнотекстовый, семантический и гибридный режимы
        </p>
      </div>

      <Card className="p-4">
        <form
          onSubmit={(event) => {
            event.preventDefault();
            run();
          }}
          className="space-y-3"
        >
          <div className="relative">
            <Search className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
            <input
              className="input !py-3 pl-10 pr-28 text-sm"
              placeholder="Например: неустойка за просрочку поставки оборудования"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />
            <button
              type="submit"
              className="btn-primary btn-sm absolute right-1.5 top-1/2 -translate-y-1/2"
              disabled={search.isPending || !query.trim()}
            >
              {search.isPending ? <Spinner className="h-3.5 w-3.5" /> : <Search className="h-3.5 w-3.5" />}
              Найти
            </button>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <div className="flex items-center gap-1 rounded-xl border border-white/10 bg-ink-900/60 p-1">
              {MODES.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => setMode(item.id)}
                  title={item.hint}
                  className={cn(
                    "flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs font-medium transition",
                    mode === item.id
                      ? "bg-brand-600/25 text-brand-100"
                      : "text-slate-400 hover:text-slate-200",
                  )}
                >
                  <item.icon className="h-3.5 w-3.5" /> {item.label}
                </button>
              ))}
            </div>

            <select
              className="input !w-auto !py-2 text-xs"
              value={docType}
              onChange={(event) => setDocType(event.target.value)}
            >
              <option value="">Все типы</option>
              {DOC_TYPES.map((type) => (
                <option key={type} value={type}>
                  {type}
                </option>
              ))}
            </select>

            <select
              className="input !w-auto !py-2 text-xs"
              value={status}
              onChange={(event) => setStatus(event.target.value)}
            >
              <option value="">Все статусы</option>
              <option value="completed">Обработан</option>
              <option value="verified">Проверен</option>
              <option value="failed">Ошибка</option>
            </select>

            <button
              type="button"
              className="btn-ghost btn-sm ml-auto"
              disabled={!result || exporting}
              onClick={async () => {
                setExporting(true);
                try {
                  await downloadFile(searchApi.exportBatchUrl(query, docType || undefined));
                  toast.success("Сводная таблица по результатам сформирована");
                } catch (error) {
                  toast.error("Экспорт не удался", String(error));
                } finally {
                  setExporting(false);
                }
              }}
            >
              <Download className="h-3.5 w-3.5" /> XLSX по результатам
            </button>
          </div>
        </form>

        <div className="mt-3 flex flex-wrap gap-1.5">
          <span className="text-[11px] text-slate-500">Примеры:</span>
          {EXAMPLES.map((example) => (
            <button
              key={example}
              onClick={() => run(example)}
              className="rounded-full border border-white/10 bg-white/[0.03] px-2.5 py-1 text-[11px] text-slate-400 transition hover:border-brand-500/40 hover:text-white"
            >
              {example}
            </button>
          ))}
        </div>
      </Card>

      {search.isError && <ErrorBanner message={apiError(search.error)} />}

      {result && (
        <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-slate-400">
          <span>
            Найдено {result.total}{" "}
            {pluralize(result.total, ["фрагмент", "фрагмента", "фрагментов"])} за {result.took_ms} мс · режим{" "}
            <span className="text-slate-200">{result.mode}</span>
          </span>
          <span>запрос: «{result.query}»</span>
        </div>
      )}

      <div className="space-y-3">
        {result?.hits.map((hit, index) => (
          <Card key={`${hit.document_id}-${hit.page_number}-${index}`} className="p-4 transition hover:border-brand-500/25">
            <div className="flex flex-wrap items-start justify-between gap-2">
              <div className="min-w-0">
                <Link
                  to={`/documents/${hit.document_id}`}
                  className="truncate text-sm font-semibold text-slate-100 hover:text-brand-200"
                >
                  {hit.filename}
                </Link>
                <div className="mt-1 flex flex-wrap items-center gap-2 text-[11px] text-slate-500">
                  {hit.page_number && <span>стр. {hit.page_number}</span>}
                  <StatusBadge status={hit.status} />
                  {hit.doc_type && <Badge tone="muted">{hit.doc_type}</Badge>}
                  <Badge tone="muted">{hit.match_type}</Badge>
                </div>
              </div>
              <div className="flex w-28 shrink-0 flex-col items-end gap-1">
                <span className="text-[10px] text-slate-500">релевантность</span>
                <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/[0.06]">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-brand-600 to-brand-300"
                    style={{ width: `${Math.min(hit.score * 100, 100)}%` }}
                  />
                </div>
                <span className="text-[10px] text-slate-400">{hit.score.toFixed(2)}</span>
              </div>
            </div>
            <p
              className="mt-2.5 rounded-lg border border-white/[0.05] bg-black/20 px-3 py-2 text-[12.5px] leading-relaxed text-slate-300"
              dangerouslySetInnerHTML={{ __html: highlight(hit.snippet, result.query) }}
            />
          </Card>
        ))}

        {result && result.hits.length === 0 && (
          <Card>
            <EmptyState
              icon={<FileSearch className="h-5 w-5" />}
              title="Ничего не найдено"
              description="Попробуйте другую формулировку или переключите режим на семантический."
            />
          </Card>
        )}

        {!result && !search.isPending && (
          <Card>
            <EmptyState
              icon={<Sparkles className="h-5 w-5" />}
              title="Введите запрос"
              description="Гибридный режим объединяет точные совпадения и векторную близость эмбеддингов."
            />
          </Card>
        )}
      </div>
    </div>
  );
}
