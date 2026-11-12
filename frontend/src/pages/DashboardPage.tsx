import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import {
  ArrowUpRight,
  BadgeCheck,
  Files,
  Layers,
  ScanLine,
  Sigma,
  Upload,
} from "lucide-react";
import { systemApi, documentsApi } from "@/api/endpoints";
import { Card, EmptyState, Progress, SectionTitle, Skeleton, StatusBadge, Badge } from "@/components/ui";
import { UploadDropzone } from "@/components/upload/UploadDropzone";
import { cn, formatDate, isProcessing, pluralize } from "@/lib/utils";
import type { DashboardStats, DocumentSummary } from "@/types";

function StatCard({
  label,
  value,
  hint,
  icon: Icon,
  accent,
}: {
  label: string;
  value: string;
  hint?: string;
  icon: typeof Files;
  accent: string;
}) {
  return (
    <Card className="p-4">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-[11px] font-medium uppercase tracking-wide text-slate-400">{label}</p>
          <p className="mt-1.5 text-2xl font-semibold tracking-tight text-white">{value}</p>
          {hint && <p className="mt-1 text-[11px] text-slate-500">{hint}</p>}
        </div>
        <div className={cn("grid h-9 w-9 place-items-center rounded-xl ring-1 ring-inset", accent)}>
          <Icon className="h-4 w-4" />
        </div>
      </div>
    </Card>
  );
}

function DocumentRow({ document }: { document: DocumentSummary }) {
  return (
    <Link
      to={`/documents/${document.id}`}
      className="group flex items-center gap-3 border-b border-white/[0.04] px-4 py-3 transition last:border-0 hover:bg-white/[0.025]"
    >
      <div className="grid h-9 w-9 shrink-0 place-items-center rounded-lg bg-white/[0.04] text-[10px] font-semibold uppercase text-slate-300">
        {(document.filename.split(".").pop() ?? "").slice(0, 4)}
      </div>
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-medium text-slate-100 group-hover:text-white">{document.filename}</p>
        <div className="mt-0.5 flex flex-wrap items-center gap-x-2 gap-y-1 text-[11px] text-slate-500">
          <span>{formatDate(document.created_at)}</span>
          <span>·</span>
          <span>
            {document.page_count} {pluralize(document.page_count, ["страница", "страницы", "страниц"])}
          </span>
          {document.doc_type && (
            <>
              <span>·</span>
              <span className="text-slate-400">{document.doc_type}</span>
            </>
          )}
        </div>
        {isProcessing(document.status) && <Progress value={document.progress} className="mt-1.5 max-w-[220px]" />}
      </div>
      <div className="flex shrink-0 items-center gap-2">
        {document.is_verified && <BadgeCheck className="h-4 w-4 text-brand-300" />}
        <StatusBadge status={document.status} />
        <ArrowUpRight className="h-4 w-4 text-slate-600 transition group-hover:text-brand-300" />
      </div>
    </Link>
  );
}

function TypeDistribution({ stats }: { stats?: DashboardStats }) {
  const entries = Object.entries(stats?.by_type ?? {}).sort((a, b) => b[1] - a[1]);
  const total = entries.reduce((sum, [, count]) => sum + count, 0) || 1;
  const palette = ["bg-brand-500", "bg-indigo-400", "bg-emerald-400", "bg-amber-400", "bg-sky-400", "bg-rose-400"];

  return (
    <Card className="p-4">
      <SectionTitle title="Типы документов" description="Результат автоматической классификации" />
      <div className="mt-4 space-y-3">
        {entries.length === 0 && <p className="text-xs text-slate-500">Пока нет обработанных документов</p>}
        {entries.map(([type, count], index) => (
          <div key={type}>
            <div className="mb-1 flex items-center justify-between text-[11px]">
              <span className="text-slate-300">{type}</span>
              <span className="text-slate-500">{count}</span>
            </div>
            <div className="h-1.5 overflow-hidden rounded-full bg-white/[0.06]">
              <div
                className={cn("h-full rounded-full", palette[index % palette.length])}
                style={{ width: `${(count / total) * 100}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
}

function WeekChart({ stats }: { stats?: DashboardStats }) {
  const data = stats?.last_7_days ?? [];
  const max = Math.max(...data.map((item) => item.count), 1);
  return (
    <Card className="p-4">
      <SectionTitle title="Загрузки за период" description="Последние 7 активных дней" />
      <div className="mt-5 flex h-28 items-end gap-2">
        {data.length === 0 && <p className="text-xs text-slate-500">Нет данных</p>}
        {data.map((item) => (
          <div key={item.date} className="flex flex-1 flex-col items-center gap-1.5">
            <div className="flex h-full w-full items-end">
              <div
                className="w-full rounded-t-md bg-gradient-to-t from-brand-700/50 to-brand-400"
                style={{ height: `${Math.max((item.count / max) * 100, 4)}%` }}
                title={`${item.date}: ${item.count}`}
              />
            </div>
            <span className="text-[10px] text-slate-500">{item.date.slice(5).replace("-", ".")}</span>
          </div>
        ))}
      </div>
    </Card>
  );
}

export function DashboardPage() {
  const statsQuery = useQuery({ queryKey: ["stats"], queryFn: systemApi.stats });
  const documentsQuery = useQuery({
    queryKey: ["documents", { page: 1, limit: 6 }],
    queryFn: () => documentsApi.list({ page: 1, limit: 6 }),
    refetchInterval: (query) =>
      query.state.data?.items.some((item) => isProcessing(item.status)) ? 2500 : false,
  });
  const statusQuery = useQuery({ queryKey: ["system-status"], queryFn: systemApi.status, retry: false });

  const stats = statsQuery.data;
  const documents = documentsQuery.data;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-white">Рабочий стол</h1>
          <p className="mt-1 text-sm text-slate-400">
            Обзор обработки документов и состояние локального ИИ-стека
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge tone={statusQuery.data?.llm.available ? "brand" : "muted"}>
            LLM: {statusQuery.data?.llm.available ? statusQuery.data.llm.model : "эвристический fallback"}
          </Badge>
          <Link to="/upload" className="btn-primary btn-sm">
            <Upload className="h-3.5 w-3.5" /> Загрузить документы
          </Link>
        </div>
      </div>

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        {statsQuery.isLoading && Array.from({ length: 4 }).map((_, index) => <Skeleton key={index} className="h-[104px]" />)}
        {stats && (
          <>
            <StatCard
              label="Всего документов"
              value={String(stats.total_documents)}
              hint={`${stats.by_status.completed ?? 0} обработано · ${stats.by_status.failed ?? 0} с ошибкой`}
              icon={Files}
              accent="bg-brand-500/15 text-brand-300 ring-brand-500/25"
            />
            <StatCard
              label="Страниц оцифровано"
              value={String(stats.pages_processed)}
              hint="включая нативный текстовый слой"
              icon={Layers}
              accent="bg-indigo-500/15 text-indigo-300 ring-indigo-500/25"
            />
            <StatCard
              label="Проверено оператором"
              value={String(stats.verified_count)}
              hint="подтверждено в режиме Human-in-the-loop"
              icon={BadgeCheck}
              accent="bg-emerald-500/15 text-emerald-300 ring-emerald-500/25"
            />
            <StatCard
              label="Средняя уверенность"
              value={`${(stats.avg_confidence * 100).toFixed(1)}%`}
              hint="по всем страницам документов"
              icon={Sigma}
              accent="bg-amber-500/15 text-amber-300 ring-amber-500/25"
            />
          </>
        )}
      </div>

      <div className="grid gap-4 xl:grid-cols-[1.6fr,1fr]">
        <div className="space-y-4">
          <Card className="p-4">
            <SectionTitle
              title="Быстрая загрузка"
              description="Пакетная загрузка: PDF, сканы, DOCX, TXT"
              action={
                <Link to="/upload" className="text-xs text-brand-300 hover:text-brand-200">
                  Расширенная загрузка →
                </Link>
              }
            />
            <div className="mt-3">
              <UploadDropzone compact />
            </div>
          </Card>

          <Card className="overflow-hidden">
            <div className="flex items-center justify-between px-4 py-3.5">
              <SectionTitle title="Последние документы" />
              <Link to="/documents" className="text-xs text-brand-300 hover:text-brand-200">
                Все документы →
              </Link>
            </div>
            {documentsQuery.isLoading && (
              <div className="space-y-2 px-4 pb-4">
                {Array.from({ length: 4 }).map((_, index) => (
                  <Skeleton key={index} className="h-12" />
                ))}
              </div>
            )}
            {documents && documents.items.length === 0 && (
              <EmptyState
                icon={<ScanLine className="h-5 w-5" />}
                title="Документов ещё нет"
                description="Загрузите первый документ — пайплайн OCR и локального ИИ запустится автоматически."
              />
            )}
            {documents?.items.map((document) => <DocumentRow key={document.id} document={document} />)}
          </Card>
        </div>

        <div className="space-y-4">
          <TypeDistribution stats={stats} />
          <WeekChart stats={stats} />
          <Card className="p-4">
            <SectionTitle title="Пайплайн" description="Этапы обработки одного документа" />
            <ol className="mt-3 space-y-2.5 text-xs">
              {[
                "Загрузка и валидация (Magic Bytes, лимит 50 МБ)",
                "Хранение оригинала + запись в БД (PENDING)",
                "Препроцессинг CV и OCR с координатами",
                "Локальная LLM: коррекция, NER, резюме",
                "Индексация (FTS + эмбеддинги)",
                "Ручная верификация и экспорт",
              ].map((step, index) => (
                <li key={step} className="flex gap-2.5">
                  <span className="mt-0.5 grid h-5 w-5 shrink-0 place-items-center rounded-full border border-brand-500/30 bg-brand-500/10 text-[10px] font-semibold text-brand-200">
                    {index + 1}
                  </span>
                  <span className="text-slate-400">{step}</span>
                </li>
              ))}
            </ol>
          </Card>
        </div>
      </div>
    </div>
  );
}
