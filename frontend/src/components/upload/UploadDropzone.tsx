import { useCallback, useState } from "react";
import { useDropzone, type FileRejection } from "react-dropzone";
import { useMutation } from "@tanstack/react-query";
import { CheckCircle2, FileText, UploadCloud } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { apiError } from "@/api/client";
import { documentsApi } from "@/api/endpoints";
import { toast } from "@/store/toast";
import { cn, formatBytes } from "@/lib/utils";
import { Progress, Spinner } from "@/components/ui";

const ACCEPT = {
  "application/pdf": [".pdf"],
  "image/*": [".png", ".jpg", ".jpeg", ".tif", ".tiff"],
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
  "text/plain": [".txt", ".md"],
};

const MAX_MB = 50;

export function UploadDropzone({ compact = false }: { compact?: boolean }) {
  const navigate = useNavigate();
  const [files, setFiles] = useState<File[]>([]);
  const [progress, setProgress] = useState(0);
  const [rejected, setRejected] = useState<string[]>([]);

  const upload = useMutation({
    mutationFn: (payload: File[]) => documentsApi.upload(payload, setProgress),
    onSuccess: (data) => {
      toast.success(
        "Документы загружены",
        `${data.documents.length} файл(ов) отправлено на обработку. Статус обновится автоматически.`,
      );
      setFiles([]);
      setProgress(0);
      if (data.documents.length === 1) {
        navigate(`/documents/${data.documents[0].id}`);
      } else {
        navigate("/documents");
      }
    },
    onError: (error) => {
      setProgress(0);
      toast.error("Не удалось загрузить файлы", apiError(error));
    },
  });

  const onDrop = useCallback((accepted: File[], rejections: FileRejection[]) => {
    setRejected(rejections.map((item) => `${item.file.name}: ${item.errors[0]?.message ?? "недопустимый файл"}`));
    setFiles((prev) => {
      const names = new Set(prev.map((file) => file.name));
      return [...prev, ...accepted.filter((file) => !names.has(file.name))];
    });
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPT,
    maxSize: MAX_MB * 1024 * 1024,
    multiple: true,
  });

  return (
    <div className="space-y-3">
      <div
        {...getRootProps()}
        className={cn(
          "group relative cursor-pointer rounded-2xl border border-dashed transition",
          isDragActive
            ? "border-brand-400 bg-brand-500/10"
            : "border-white/15 bg-white/[0.02] hover:border-brand-500/50 hover:bg-white/[0.04]",
          compact ? "px-5 py-6" : "px-6 py-10",
        )}
      >
        <input {...getInputProps()} />
        <div className={cn("flex items-center gap-4", compact ? "" : "flex-col text-center")}>
          <div className="grid h-11 w-11 shrink-0 place-items-center rounded-xl bg-gradient-to-br from-brand-500/30 to-brand-700/20 text-brand-200 ring-1 ring-inset ring-brand-500/30">
            <UploadCloud className="h-5 w-5" />
          </div>
          <div className={compact ? "text-left" : ""}>
            <p className="text-sm font-semibold text-slate-100">
              {isDragActive ? "Отпустите файлы для загрузки" : "Перетащите документы или выберите файлы"}
            </p>
            <p className="mt-1 text-xs text-slate-400">
              PDF, JPG/PNG/TIFF, DOCX, TXT · до {MAX_MB} МБ · проверка Magic Bytes на сервере
            </p>
          </div>
        </div>
      </div>

      {rejected.length > 0 && (
        <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-200">
          {rejected.map((message) => (
            <div key={message}>{message}</div>
          ))}
        </div>
      )}

      {files.length > 0 && (
        <div className="space-y-2 rounded-xl border border-white/[0.07] bg-ink-900/60 p-3">
          {files.map((file) => (
            <div key={file.name} className="flex items-center gap-2 text-xs text-slate-300">
              <FileText className="h-3.5 w-3.5 shrink-0 text-brand-300" />
              <span className="min-w-0 flex-1 truncate">{file.name}</span>
              <span className="text-slate-500">{formatBytes(file.size)}</span>
            </div>
          ))}

          {upload.isPending && (
            <div className="space-y-1 pt-1">
              <Progress value={progress} />
              <p className="text-[11px] text-slate-400">
                {progress < 100 ? `Отправка… ${progress}%` : "Файлы отправлены, идёт постановка в очередь…"}
              </p>
            </div>
          )}

          <div className="flex items-center justify-between gap-2 pt-1">
            <button
              type="button"
              className="text-xs text-slate-400 underline-offset-2 hover:text-slate-200 hover:underline"
              onClick={() => setFiles([])}
              disabled={upload.isPending}
            >
              Очистить
            </button>
            <button
              type="button"
              className="btn-primary btn-sm"
              disabled={upload.isPending}
              onClick={() => upload.mutate(files)}
            >
              {upload.isPending ? <Spinner className="mr-1 h-3.5 w-3.5" /> : <CheckCircle2 className="h-3.5 w-3.5" />}
              Загрузить {files.length}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
