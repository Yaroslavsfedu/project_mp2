import { useState } from "react";
import { Navigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Cpu,
  Database,
  Gauge,
  Play,
  RefreshCw,
  ScrollText,
  Server,
  ShieldAlert,
  Users,
  XCircle,
} from "lucide-react";
import { adminApi, systemApi } from "@/api/endpoints";
import { apiError } from "@/api/client";
import { toast } from "@/store/toast";
import { useAuth } from "@/store/auth";
import { Badge, Card, EmptyState, ErrorBanner, Skeleton } from "@/components/ui";
import { cn, formatBytes, formatDate } from "@/lib/utils";
import type { AdminUser, QualityReport } from "@/types";

type Tab = "users" | "audit" | "quality" | "system";

const TABS: { id: Tab; label: string; icon: typeof Users }[] = [
  { id: "users", label: "Пользователи", icon: Users },
  { id: "audit", label: "Журнал действий", icon: ScrollText },
  { id: "quality", label: "Качество ИИ", icon: Gauge },
  { id: "system", label: "Система и GPU", icon: Server },
];

const ROLE_LABELS: Record<string, string> = {
  admin: "Администратор",
  operator: "Оператор",
  viewer: "Наблюдатель",
};

const ACTION_LABELS: Record<string, string> = {
  document_uploaded: "Загрузка документа",
  document_verified: "Верификация",
  document_edited: "Изменение данных",
  document_deleted: "Удаление",
  document_reprocessed: "Повторная обработка",
  user_updated: "Изменение пользователя",
  quality_evaluated: "Оценка качества",
};

const percent = (value: number | undefined, digits = 2) =>
  value === undefined || Number.isNaN(value) ? "—" : `${(value * 100).toFixed(digits)} %`;

export function AdminPage() {
  const user = useAuth((state) => state.user);
  const [tab, setTab] = useState<Tab>("users");

  if (user && user.role !== "admin") {
    return <Navigate to="/" replace />;
  }

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-white">Администрирование</h1>
        <p className="mt-1 text-sm text-slate-400">
          Пользователи и роли, журнал аудита, метрики качества моделей и состояние локального ИИ-стека
        </p>
      </div>

      <div className="flex flex-wrap gap-1.5">
        {TABS.map((item) => (
          <button
            key={item.id}
            onClick={() => setTab(item.id)}
            className={cn(
              "inline-flex items-center gap-2 rounded-xl border px-3.5 py-2 text-xs font-medium transition",
              tab === item.id
                ? "border-brand-500/40 bg-brand-500/15 text-brand-100"
                : "border-white/10 bg-white/[0.02] text-slate-400 hover:text-slate-200",
            )}
          >
            <item.icon className="h-3.5 w-3.5" />
            {item.label}
          </button>
        ))}
      </div>

      {tab === "users" && <UsersTab />}
      {tab === "audit" && <AuditTab />}
      {tab === "quality" && <QualityTab />}
      {tab === "system" && <SystemTab />}
    </div>
  );
}

function StatTile({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <Card className="p-4">
      <p className="text-[11px] font-medium uppercase tracking-wide text-slate-500">{label}</p>
      <p className="mt-1.5 text-xl font-semibold text-white">{value}</p>
      {hint && <p className="mt-0.5 text-[11px] text-slate-500">{hint}</p>}
    </Card>
  );
}

function UsersTab() {
  const queryClient = useQueryClient();
  const overview = useQuery({ queryKey: ["admin-overview"], queryFn: adminApi.overview });
  const users = useQuery({ queryKey: ["admin-users"], queryFn: adminApi.users });

  const update = useMutation({
    mutationFn: ({ id, payload }: { id: string; payload: { role?: string; is_active?: boolean } }) =>
      adminApi.updateUser(id, payload),
    onSuccess: (data) => {
      toast.success("Пользователь обновлён", data.email);
      queryClient.invalidateQueries({ queryKey: ["admin-users"] });
      queryClient.invalidateQueries({ queryKey: ["admin-overview"] });
      queryClient.invalidateQueries({ queryKey: ["admin-audit"] });
    },
    onError: (error) => toast.error("Не удалось обновить", apiError(error)),
  });

  const data = overview.data;

  return (
    <div className="space-y-4">
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        {!data && Array.from({ length: 4 }).map((_, index) => <Skeleton key={index} className="h-[92px]" />)}
        {data && (
          <>
            <StatTile label="Пользователи" value={String(data.users_total)} hint="Активные учётные записи" />
            <StatTile
              label="Документы"
              value={String(data.documents_total)}
              hint={`${data.pages_total} страниц · ${data.verified_total} проверено`}
            />
            <StatTile label="Объём файлов" value={formatBytes(data.storage_bytes)} hint="В локальном хранилище" />
            <StatTile
              label="События аудита"
              value={String(data.audit_events)}
              hint={data.actions_24h.map((item) => `${ACTION_LABELS[item.action] ?? item.action}: ${item.count}`).join(" · ") || "за 24 часа — нет"}
            />
          </>
        )}
      </div>

      <Card className="overflow-hidden">
        <div className="flex items-center justify-between border-b border-white/[0.06] px-4 py-3">
          <p className="text-sm font-medium text-slate-200">Учётные записи и роли</p>
          <button
            className="btn-ghost btn-sm"
            onClick={() => queryClient.invalidateQueries({ queryKey: ["admin-users"] })}
          >
            <RefreshCw className="h-3.5 w-3.5" /> Обновить
          </button>
        </div>

        {users.isLoading && <div className="p-4"><Skeleton className="h-32" /></div>}
        {users.error && <div className="p-4"><ErrorBanner message={apiError(users.error)} /></div>}

        {users.data && (
          <table className="w-full text-sm">
            <thead className="bg-white/[0.02] text-[11px] uppercase tracking-wide text-slate-500">
              <tr>
                <th className="px-4 py-2.5 text-left font-medium">Пользователь</th>
                <th className="px-4 py-2.5 text-left font-medium">Роль</th>
                <th className="px-4 py-2.5 text-left font-medium">Документы</th>
                <th className="px-4 py-2.5 text-left font-medium">Создан</th>
                <th className="px-4 py-2.5 text-right font-medium">Доступ</th>
              </tr>
            </thead>
            <tbody>
              {users.data.map((item) => (
                <tr key={item.id} className="border-t border-white/[0.04]">
                  <td className="px-4 py-3">
                    <p className="font-medium text-slate-200">{item.full_name || "—"}</p>
                    <p className="text-[11px] text-slate-500">{item.email}</p>
                  </td>
                  <td className="px-4 py-3">
                    <select
                      className="input !py-1.5 !text-xs"
                      value={item.role}
                      disabled={update.isPending}
                      onChange={(event) =>
                        update.mutate({ id: item.id, payload: { role: event.target.value } })
                      }
                    >
                      {Object.entries(ROLE_LABELS).map(([value, label]) => (
                        <option key={value} value={value}>
                          {label}
                        </option>
                      ))}
                    </select>
                  </td>
                  <td className="px-4 py-3 text-slate-300">{item.documents_count}</td>
                  <td className="px-4 py-3 text-[12px] text-slate-400">{formatDate(item.created_at)}</td>
                  <td className="px-4 py-3 text-right">
                    <button
                      disabled={update.isPending}
                      onClick={() =>
                        update.mutate({ id: item.id, payload: { is_active: !item.is_active } })
                      }
                      className={cn(
                        "chip",
                        item.is_active
                          ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-300"
                          : "border-rose-500/30 bg-rose-500/10 text-rose-300",
                      )}
                    >
                      {item.is_active ? <CheckCircle2 className="h-3 w-3" /> : <XCircle className="h-3 w-3" />}
                      {item.is_active ? "Активен" : "Заблокирован"}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>
    </div>
  );
}

function AuditTab() {
  const [action, setAction] = useState("");
  const queryClient = useQueryClient();
  const audit = useQuery({
    queryKey: ["admin-audit", action],
    queryFn: () => adminApi.audit({ limit: 200, ...(action ? { action } : {}) }),
  });

  return (
    <Card className="overflow-hidden">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-white/[0.06] px-4 py-3">
        <p className="text-sm font-medium text-slate-200">Журнал действий (последние 200 событий)</p>
        <div className="flex items-center gap-2">
          <select className="input !py-1.5 !text-xs" value={action} onChange={(event) => setAction(event.target.value)}>
            <option value="">Все действия</option>
            {Object.entries(ACTION_LABELS).map(([value, label]) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
          </select>
          <button
            className="btn-ghost btn-sm"
            onClick={() => queryClient.invalidateQueries({ queryKey: ["admin-audit"] })}
          >
            <RefreshCw className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {audit.isLoading && <div className="p-4"><Skeleton className="h-40" /></div>}
      {audit.error && <div className="p-4"><ErrorBanner message={apiError(audit.error)} /></div>}
      {audit.data?.length === 0 && (
        <div className="p-6">
          <EmptyState
            icon={<ScrollText className="h-5 w-5" />}
            title="Событий пока нет"
            description="Загрузите документ, подтвердите данные или измените роль — все действия попадут в журнал."
          />
        </div>
      )}
      {audit.data && audit.data.length > 0 && (
        <div className="max-h-[560px] overflow-auto">
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-ink-900 text-[11px] uppercase tracking-wide text-slate-500">
              <tr>
                <th className="px-4 py-2.5 text-left font-medium">Время</th>
                <th className="px-4 py-2.5 text-left font-medium">Действие</th>
                <th className="px-4 py-2.5 text-left font-medium">Пользователь</th>
                <th className="px-4 py-2.5 text-left font-medium">Документ</th>
              </tr>
            </thead>
            <tbody>
              {audit.data.map((entry) => (
                <tr key={entry.id} className="border-t border-white/[0.04]">
                  <td className="whitespace-nowrap px-4 py-2.5 text-[12px] text-slate-400">
                    {formatDate(entry.created_at)}
                  </td>
                  <td className="px-4 py-2.5">
                    <Badge tone="muted" className="text-[10px]">
                      {ACTION_LABELS[entry.action] ?? entry.action}
                    </Badge>
                  </td>
                  <td className="px-4 py-2.5 text-[12px] text-slate-300">{entry.user_email}</td>
                  <td className="max-w-[280px] truncate px-4 py-2.5 text-[12px] text-slate-400">
                    {entry.document_name || "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}

function MetricCard({
  label,
  value,
  hint,
  tone = "default",
}: {
  label: string;
  value: string;
  hint?: string;
  tone?: "default" | "good" | "warn";
}) {
  return (
    <Card className="p-4">
      <p className="text-[11px] font-medium uppercase tracking-wide text-slate-500">{label}</p>
      <p
        className={cn(
          "mt-1.5 text-2xl font-semibold tabular-nums",
          tone === "good" && "text-emerald-300",
          tone === "warn" && "text-amber-300",
          tone === "default" && "text-white",
        )}
      >
        {value}
      </p>
      {hint && <p className="mt-0.5 text-[11px] text-slate-500">{hint}</p>}
    </Card>
  );
}

function QualityTab() {
  const queryClient = useQueryClient();
  const quality = useQuery({
    queryKey: ["admin-quality"],
    queryFn: adminApi.quality,
    retry: false,
  });
  const run = useMutation({
    mutationFn: () => adminApi.runQuality(),
    onSuccess: (data) => {
      queryClient.setQueryData(["admin-quality"], { report: data.report, markdown: data.markdown });
      toast.success("Оценка завершена", `CER ${percent(data.report.metrics.cer)} · F1 ${percent(data.report.metrics.ner_f1, 1)}`);
    },
    onError: (error) => toast.error("Оценка не выполнена", apiError(error)),
  });

  const report: QualityReport | undefined = quality.data?.report;
  const notFound = (quality.error as { response?: { status?: number } } | null)?.response?.status === 404;

  return (
    <div className="space-y-4">
      <Card className="flex flex-wrap items-center justify-between gap-3 p-4">
        <div>
          <p className="text-sm font-medium text-slate-200">Эталонный датасет: CER · WER · NER F1</p>
          <p className="mt-0.5 text-[12px] text-slate-500">
            Прогон выполняется в изолированной временной базе — рабочие документы не затрагиваются.
            {report && ` Последний отчёт: ${formatDate(report.generated_at)}.`}
          </p>
        </div>
        <button className="btn-primary btn-sm" onClick={() => run.mutate()} disabled={run.isPending}>
          {run.isPending ? <RefreshCw className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
          {run.isPending ? "Оценка выполняется…" : "Запустить оценку"}
        </button>
      </Card>

      {quality.isLoading && <Skeleton className="h-52" />}
      {quality.error && !notFound && <ErrorBanner message={apiError(quality.error)} />}
      {notFound && (
        <EmptyState
          icon={<Gauge className="h-5 w-5" />}
          title="Отчёт ещё не сформирован"
          description="Нажмите «Запустить оценку»: платформа обработает samples/ground_truth.json и посчитает метрики."
        />
      )}

      {report && (
        <>
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
            <MetricCard
              label="CER"
              value={percent(report.metrics.cer)}
              hint="Доля ошибочных символов"
              tone={report.metrics.cer < 0.05 ? "good" : "warn"}
            />
            <MetricCard
              label="WER"
              value={percent(report.metrics.wer)}
              hint="Доля ошибочных слов"
              tone={report.metrics.wer < 0.1 ? "good" : "warn"}
            />
            <MetricCard
              label="NER F1"
              value={percent(report.metrics.ner_f1, 1)}
              hint={`P ${percent(report.metrics.ner_precision, 1)} · R ${percent(report.metrics.ner_recall, 1)}`}
              tone={report.metrics.ner_f1 > 0.85 ? "good" : "warn"}
            />
            <MetricCard
              label="Тип документа"
              value={percent(report.metrics.doc_type_accuracy, 1)}
              hint="Точность классификации"
              tone={report.metrics.doc_type_accuracy > 0.9 ? "good" : "warn"}
            />
            <MetricCard
              label="Время обработки"
              value={`${report.metrics.avg_duration_sec.toFixed(2)} с`}
              hint={`${report.documents} документов в выборке`}
            />
          </div>

          {report.skipped.length > 0 && (
            <Card className="flex items-start gap-2 border-amber-500/25 bg-amber-500/[0.06] p-3.5 text-[12px] text-amber-200">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
              <div>
                Пропущено документов: {report.skipped.length} — требуется OCR-движок (Tesseract или
                локальная VLM). Установите Tesseract с языком rus или загрузите{" "}
                <code className="rounded bg-black/30 px-1">qwen2-vl:7b</code> и повторите оценку —
                тогда в метрики попадут реальные сканы.
              </div>
            </Card>
          )}

          <Card className="overflow-hidden">
            <div className="border-b border-white/[0.06] px-4 py-3 text-sm font-medium text-slate-200">
              Документы выборки
            </div>
            <table className="w-full text-sm">
              <thead className="bg-white/[0.02] text-[11px] uppercase tracking-wide text-slate-500">
                <tr>
                  <th className="px-4 py-2.5 text-left font-medium">Документ</th>
                  <th className="px-4 py-2.5 text-left font-medium">Тип</th>
                  <th className="px-4 py-2.5 text-right font-medium">CER</th>
                  <th className="px-4 py-2.5 text-right font-medium">WER</th>
                  <th className="px-4 py-2.5 text-right font-medium">Время</th>
                  <th className="px-4 py-2.5 text-right font-medium">Источник</th>
                </tr>
              </thead>
              <tbody>
                {report.per_document.map((row) => (
                  <tr key={row.filename} className="border-t border-white/[0.04]">
                    <td className="px-4 py-2.5 text-slate-200">{row.filename}</td>
                    <td className="px-4 py-2.5 text-[12px] text-slate-400">
                      {row.doc_type_predicted || "—"}
                      {!row.doc_type_correct && <span className="ml-1 text-rose-400">≠</span>}
                    </td>
                    <td className="px-4 py-2.5 text-right tabular-nums text-slate-300">{percent(row.cer)}</td>
                    <td className="px-4 py-2.5 text-right tabular-nums text-slate-300">{percent(row.wer)}</td>
                    <td className="px-4 py-2.5 text-right tabular-nums text-slate-400">
                      {row.duration_sec.toFixed(2)} с
                    </td>
                    <td className="px-4 py-2.5 text-right text-[11px] text-slate-500">
                      {row.ocr_source === "native" ? "текстовый слой" : row.ocr_source}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          {quality.data?.markdown && (
            <details className="card p-4">
              <summary className="cursor-pointer text-sm font-medium text-slate-200">
                Markdown-отчёт (для отчёта по НИР)
              </summary>
              <pre className="mt-3 max-h-96 overflow-auto whitespace-pre-wrap rounded-lg bg-black/30 p-3 text-[11px] leading-relaxed text-slate-300">
                {quality.data.markdown}
              </pre>
            </details>
          )}
        </>
      )}
    </div>
  );
}

function StatusRow({ label, ok, value }: { label: string; ok: boolean; value: string }) {
  return (
    <div className="flex items-center justify-between gap-3 border-t border-white/[0.04] py-2 text-[12px] first:border-t-0">
      <span className="text-slate-400">{label}</span>
      <span className={cn("flex items-center gap-1.5 font-medium", ok ? "text-emerald-300" : "text-amber-300")}>
        {ok ? <CheckCircle2 className="h-3.5 w-3.5" /> : <AlertTriangle className="h-3.5 w-3.5" />}
        {value}
      </span>
    </div>
  );
}

function SystemTab() {
  const status = useQuery({ queryKey: ["system-status"], queryFn: systemApi.status, retry: false });
  const data = status.data;

  if (status.isLoading) return <Skeleton className="h-64" />;
  if (status.error) return <ErrorBanner message={apiError(status.error)} />;
  if (!data) return null;

  const gpu = data.gpu;

  return (
    <div className="space-y-4">
      <div className="grid gap-3 lg:grid-cols-2">
        <Card className="p-4">
          <p className="mb-2 flex items-center gap-1.5 text-sm font-medium text-slate-200">
            <Cpu className="h-4 w-4 text-brand-300" /> Локальная LLM (Ollama)
          </p>
          <div className="space-y-1">
            <StatusRow label="Сервер Ollama" ok={data.llm.available} value={data.llm.available ? data.llm.url : "недоступен"} />
            <StatusRow
              label="Модель анализа"
              ok={Boolean(data.llm.model_installed)}
              value={data.llm.model}
            />
            <StatusRow
              label="Эмбеддинги"
              ok={Boolean(data.llm.embedding_model_installed)}
              value={data.llm.embedding_model}
            />
            <StatusRow
              label="VLM для OCR сканов"
              ok={Boolean(data.llm.vision_model_installed)}
              value={data.llm.vision_model ?? "—"}
            />
            <StatusRow
              label="Режим пайплайна"
              ok={data.llm.mode === "ollama"}
              value={data.llm.mode === "ollama" ? "локальная LLM" : "эвристический fallback"}
            />
          </div>
          {!!data.llm.installed_models?.length && (
            <div className="mt-3 flex flex-wrap gap-1.5">
              {data.llm.installed_models.map((model) => (
                <Badge key={model} tone="muted" className="text-[10px]">
                  {model}
                </Badge>
              ))}
            </div>
          )}
        </Card>

        <Card className="p-4">
          <p className="mb-2 flex items-center gap-1.5 text-sm font-medium text-slate-200">
            <Activity className="h-4 w-4 text-brand-300" /> GPU и OCR
          </p>
          <div className="space-y-1">
            <StatusRow
              label="Видеокарта"
              ok={gpu.available}
              value={
                gpu.available
                  ? `${gpu.name} · ${((gpu.memory_mb ?? 0) / 1024).toFixed(0)} ГБ VRAM`
                  : gpu.reason ?? "не обнаружена"
              }
            />
            <StatusRow label="OCR-движок" ok={data.ocr.engine !== "none"} value={data.ocr.engine} />
            <StatusRow label="Tesseract" ok={data.ocr.tesseract_installed} value={data.ocr.tesseract_installed ? "установлен" : "не найден"} />
            <StatusRow label="Языки OCR" ok value={data.ocr.languages} />
            <StatusRow label="Ядра CPU" ok value={String(data.cpu.cores)} />
            <StatusRow label="Очередь задач" ok value={data.queue_backend} />
            <StatusRow label="Хранилище" ok value={data.storage_backend} />
          </div>
          {gpu.available && (
            <p className="mt-3 rounded-lg border border-brand-500/25 bg-brand-500/[0.07] p-2.5 text-[11px] leading-relaxed text-brand-100">
              Рекомендуемые модели для этой карты: <b>{gpu.recommended_llm}</b> для анализа и{" "}
              <b>{gpu.recommended_vision}</b> для OCR сканов.
            </p>
          )}
        </Card>
      </div>

      <Card className="p-4">
        <p className="mb-2 flex items-center gap-1.5 text-sm font-medium text-slate-200">
          <Database className="h-4 w-4 text-brand-300" /> Что нужно донастроить
        </p>
        {data.hints.length === 0 ? (
          <p className="flex items-center gap-2 text-[12px] text-emerald-300">
            <CheckCircle2 className="h-4 w-4" /> Все компоненты локального ИИ-стека доступны.
          </p>
        ) : (
          <ul className="space-y-2">
            {data.hints.map((hint, index) => (
              <li key={index} className="flex items-start gap-2 text-[12px] leading-relaxed text-slate-300">
                <ShieldAlert className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-300" />
                {hint}
              </li>
            ))}
          </ul>
        )}
      </Card>
    </div>
  );
}
