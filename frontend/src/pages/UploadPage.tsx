import { useQuery } from "@tanstack/react-query";
import { Cpu, FileStack, ScanLine, ShieldCheck } from "lucide-react";
import { documentsApi } from "@/api/endpoints";
import { UploadDropzone } from "@/components/upload/UploadDropzone";
import { Card, Progress, SectionTitle, StatusBadge } from "@/components/ui";
import { formatDate, isProcessing } from "@/lib/utils";

export function UploadPage() {
  const documentsQuery = useQuery({
    queryKey: ["documents", { page: 1, limit: 8, sort: "created_at_desc" }],
    queryFn: () => documentsApi.list({ page: 1, limit: 8 }),
    refetchInterval: (data) =>
      data.state.data?.items.some((item) => isProcessing(item.status)) ? 2000 : false,
  });

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-white">Загрузка документов</h1>
        <p className="mt-1 text-sm text-slate-400">
          Пакетная загрузка с валидацией на клиенте и сервере, препроцессинг и постановка в очередь обработки
        </p>
      </div>

      <div className="grid gap-4 xl:grid-cols-[1.5fr,1fr]">
        <Card className="p-5">
          <SectionTitle
            title="Новые документы"
            description="Перетащите файлы или выберите вручную — до 50 МБ на файл"
          />
          <div className="mt-4">
            <UploadDropzone />
          </div>
        </Card>

        <div className="space-y-4">
          <Card className="p-4">
            <SectionTitle title="Поддерживаемые форматы" />
            <ul className="mt-3 space-y-2 text-xs text-slate-400">
              {[
                "PDF — гибридные и сканы (нативный слой используется без OCR)",
                "JPG / PNG / TIFF — фото и сканы с препроцессингом OpenCV",
                "DOCX — текст и таблицы с рендером страницы",
                "TXT / MD — быстрый импорт текстовых материалов",
              ].map((item) => (
                <li key={item} className="flex gap-2">
                  <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-brand-400" />
                  {item}
                </li>
              ))}
            </ul>
          </Card>

          <Card className="p-4">
            <SectionTitle title="Что происходит дальше" />
            <div className="mt-3 space-y-3">
              {[
                { icon: ShieldCheck, title: "Валидация", text: "Magic Bytes, расширение, лимит размера, безопасное имя" },
                { icon: ScanLine, title: "OCR + CV", text: "Deskewing, удаление печатей, бинаризация, Bounding Boxes" },
                { icon: Cpu, title: "Локальный ИИ", text: "Коррекция текста, NER, классификация, резюме, эмбеддинги" },
                { icon: FileStack, title: "Индексация", text: "Полнотекстовый и векторный поиск, готовность к верификации" },
              ].map((step) => (
                <div key={step.title} className="flex gap-2.5">
                  <div className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-white/[0.04] text-brand-300">
                    <step.icon className="h-4 w-4" />
                  </div>
                  <div>
                    <p className="text-xs font-medium text-slate-200">{step.title}</p>
                    <p className="text-[11px] leading-relaxed text-slate-500">{step.text}</p>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>
      </div>

      <Card>
        <div className="px-4 py-3.5">
          <SectionTitle title="Очередь обработки" description="Статусы обновляются через WebSocket" />
        </div>
        {(documentsQuery.data?.items ?? []).map((document) => (
          <div
            key={document.id}
            className="flex flex-wrap items-center gap-3 border-t border-white/[0.04] px-4 py-3"
          >
            <div className="min-w-0 flex-1">
              <p className="truncate text-sm text-slate-100">{document.filename}</p>
              <p className="mt-0.5 text-[11px] text-slate-500">
                {formatDate(document.created_at)} · {document.doc_type ?? "тип не определён"}
              </p>
              {isProcessing(document.status) && <Progress value={document.progress} className="mt-2 max-w-sm" />}
              {document.error_message && (
                <p className="mt-1 text-[11px] text-rose-400">{document.error_message}</p>
              )}
            </div>
            <StatusBadge status={document.status} />
          </div>
        ))}
        {(documentsQuery.data?.items ?? []).length === 0 && (
          <p className="px-4 pb-4 text-xs text-slate-500">Очередь пуста — загрузите документы выше.</p>
        )}
      </Card>
    </div>
  );
}
