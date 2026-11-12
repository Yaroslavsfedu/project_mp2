import { useEffect, useMemo, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { Plus, Save, ShieldCheck, X } from "lucide-react";
import { apiError } from "@/api/client";
import { documentsApi } from "@/api/endpoints";
import { toast } from "@/store/toast";
import { cn } from "@/lib/utils";
import { DOC_TYPES, ENTITY_LABELS, type DocumentDetail, type Entities } from "@/types";
import { Badge, Card, Spinner } from "@/components/ui";

const ARRAY_KEYS = ["persons", "organizations", "dates", "amounts", "inn_ogrn", "addresses"] as const;
type ArrayKey = (typeof ARRAY_KEYS)[number];

interface Props {
  document: DocumentDetail;
  onSaved: () => void;
}

export function EntitiesEditor({ document, onSaved }: Props) {
  const [docType, setDocType] = useState(document.doc_type ?? "Другое");
  const [summary, setSummary] = useState(document.summary ?? "");
  const [entities, setEntities] = useState(document.entities ?? ({} as Entities));
  const [keywords, setKeywords] = useState<string[]>(document.keywords ?? []);
  const [newValues, setNewValues] = useState<Record<string, string>>({});
  const [dirty, setDirty] = useState(false);

  useEffect(() => {
    setDocType(document.doc_type ?? "Другое");
    setSummary(document.summary ?? "");
    setEntities(document.entities ?? ({} as Entities));
    setKeywords(document.keywords ?? []);
    setDirty(false);
  }, [document.id, document.updated_at, document.doc_type, document.summary]);

  const save = useMutation({
    mutationFn: () =>
      documentsApi.verify(document.id, {
        document_type: docType,
        summary,
        entities,
        keywords,
        is_verified: true,
      }),
    onSuccess: () => {
      toast.success("Изменения сохранены", "Документ переведён в статус «Проверен»");
      setDirty(false);
      onSaved();
    },
    onError: (error) => toast.error("Не удалось сохранить", apiError(error)),
  });

  const update = (mutate: (draft: Entities) => void) => {
    setEntities((prev) => {
      const draft: Entities = { ...prev };
      mutate(draft);
      return draft;
    });
    setDirty(true);
  };

  const totalEntities = useMemo(
    () =>
      ARRAY_KEYS.reduce((sum, key) => sum + (entities?.[key]?.length ?? 0), 0) +
      (entities?.document_number ? 1 : 0),
    [entities],
  );

  return (
    <div className="space-y-4 pb-6">
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-white/[0.07] bg-white/[0.02] px-4 py-3">
        <div className="flex items-center gap-2 text-xs text-slate-400">
          <ShieldCheck className={cn("h-4 w-4", document.is_verified ? "text-emerald-400" : "text-slate-500")} />
          {document.is_verified ? (
            <span>Проверено оператором · {totalEntities} сущностей подтверждено</span>
          ) : (
            <span>Проверьте поля и подтвердите документ (Human-in-the-loop)</span>
          )}
        </div>
        <button
          className="btn-primary btn-sm"
          disabled={save.isPending}
          onClick={() => save.mutate()}
        >
          {save.isPending ? <Spinner className="h-3.5 w-3.5" /> : <Save className="h-3.5 w-3.5" />}
          Сохранить и подтвердить
        </button>
      </div>

      <div className="grid gap-4 md:grid-cols-[220px,1fr]">
        <div>
          <label className="label">Тип документа</label>
          <select
            className="input"
            value={docType}
            onChange={(event) => {
              setDocType(event.target.value);
              setDirty(true);
            }}
          >
            {DOC_TYPES.map((type) => (
              <option key={type} value={type}>
                {type}
              </option>
            ))}
          </select>
          <p className="mt-2 text-[11px] leading-relaxed text-slate-500">
            Определён движком <span className="text-slate-300">{document.analysis_engine ?? "—"}</span>. Можно
            скорректировать вручную.
          </p>
        </div>
        <div>
          <label className="label">Резюме документа</label>
          <textarea
            className="input min-h-[92px] resize-y leading-relaxed"
            value={summary}
            onChange={(event) => {
              setSummary(event.target.value);
              setDirty(true);
            }}
            placeholder="Краткое содержание документа…"
          />
        </div>
      </div>

      <Card className="p-4">
        <div className="mb-3 flex items-center justify-between">
          <h3 className="text-sm font-semibold text-white">Извлечённые сущности</h3>
          <Badge tone="muted">{totalEntities}</Badge>
        </div>

        <div className="space-y-4">
          {ARRAY_KEYS.map((key) => (
            <div key={key}>
              <p className="mb-1.5 text-[11px] font-medium uppercase tracking-wide text-slate-400">
                {ENTITY_LABELS[key]}
              </p>
              <div className="flex flex-wrap gap-1.5">
                {(entities?.[key] ?? []).map((value, index) => (
                  <span
                    key={`${key}-${index}`}
                    className="group inline-flex max-w-full items-center gap-1 rounded-lg border border-white/10 bg-white/[0.04] px-2 py-1 text-xs text-slate-200"
                  >
                    <span className="truncate">{value}</span>
                    <button
                      className="text-slate-500 opacity-0 transition hover:text-rose-300 group-hover:opacity-100"
                      onClick={() =>
                        update((draft) => {
                          draft[key] = draft[key].filter((_, i) => i !== index);
                        })
                      }
                    >
                      <X className="h-3 w-3" />
                    </button>
                  </span>
                ))}
                {(entities?.[key] ?? []).length === 0 && (
                  <span className="text-xs text-slate-500">не найдено</span>
                )}
              </div>
              <div className="mt-1.5 flex items-center gap-1.5">
                <input
                  className="input !py-1.5 text-xs"
                  placeholder="Добавить значение…"
                  value={newValues[key] ?? ""}
                  onChange={(event) => setNewValues((prev) => ({ ...prev, [key]: event.target.value }))}
                  onKeyDown={(event) => {
                    if (event.key !== "Enter") return;
                    const value = (newValues[key] ?? "").trim();
                    if (!value) return;
                    update((draft) => {
                      draft[key as ArrayKey] = [...(draft[key as ArrayKey] ?? []), value];
                    });
                    setNewValues((prev) => ({ ...prev, [key]: "" }));
                  }}
                />
                <button
                  className="btn-ghost btn-sm !px-2"
                  onClick={() => {
                    const value = (newValues[key] ?? "").trim();
                    if (!value) return;
                    update((draft) => {
                      draft[key as ArrayKey] = [...(draft[key as ArrayKey] ?? []), value];
                    });
                    setNewValues((prev) => ({ ...prev, [key]: "" }));
                  }}
                >
                  <Plus className="h-3.5 w-3.5" />
                </button>
              </div>
            </div>
          ))}

          <div>
            <p className="mb-1.5 text-[11px] font-medium uppercase tracking-wide text-slate-400">
              {ENTITY_LABELS.document_number}
            </p>
            <input
              className="input max-w-xs"
              value={entities?.document_number ?? ""}
              placeholder="не найден"
              onChange={(event) => {
                setEntities((prev) => ({ ...prev, document_number: event.target.value || null }));
                setDirty(true);
              }}
            />
          </div>
        </div>
      </Card>

      <Card className="p-4">
        <p className="mb-2 text-[11px] font-medium uppercase tracking-wide text-slate-400">Ключевые слова</p>
        <div className="flex flex-wrap gap-1.5">
          {keywords.map((word, index) => (
            <span
              key={`${word}-${index}`}
              className="group inline-flex items-center gap-1 rounded-full border border-brand-500/25 bg-brand-500/10 px-2.5 py-1 text-xs text-brand-100"
            >
              {word}
              <button
                className="opacity-0 transition group-hover:opacity-100"
                onClick={() => {
                  setKeywords((prev) => prev.filter((_, i) => i !== index));
                  setDirty(true);
                }}
              >
                <X className="h-3 w-3" />
              </button>
            </span>
          ))}
        </div>
        <div className="mt-2 flex items-center gap-1.5">
          <input
            className="input !py-1.5 max-w-xs text-xs"
            placeholder="Добавить тег и нажать Enter"
            value={newValues.tag ?? ""}
            onChange={(event) => setNewValues((prev) => ({ ...prev, tag: event.target.value }))}
            onKeyDown={(event) => {
              if (event.key !== "Enter") return;
              const value = (newValues.tag ?? "").trim();
              if (!value) return;
              setKeywords((prev) => [...prev, value]);
              setNewValues((prev) => ({ ...prev, tag: "" }));
              setDirty(true);
            }}
          />
        </div>
      </Card>
    </div>
  );
}
