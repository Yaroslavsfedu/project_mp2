import { CheckCircle2, Info, X, XCircle } from "lucide-react";
import { useToastStore } from "@/store/toast";
import { cn } from "@/lib/utils";

const ICONS = {
  success: CheckCircle2,
  error: XCircle,
  info: Info,
};

const TONES = {
  success: "border-emerald-500/30 bg-emerald-500/10 text-emerald-200",
  error: "border-rose-500/30 bg-rose-500/10 text-rose-200",
  info: "border-brand-500/30 bg-brand-500/10 text-brand-100",
};

export function Toaster() {
  const { toasts, dismiss } = useToastStore();
  return (
    <div className="pointer-events-none fixed bottom-6 right-6 z-[60] flex w-full max-w-sm flex-col gap-2">
      {toasts.map((item) => {
        const Icon = ICONS[item.kind];
        return (
          <div
            key={item.id}
            className={cn(
              "pointer-events-auto flex animate-fade-in items-start gap-3 rounded-xl border px-4 py-3 shadow-lg backdrop-blur",
              TONES[item.kind],
            )}
          >
            <Icon className="mt-0.5 h-4 w-4 shrink-0" />
            <div className="min-w-0 flex-1">
              <p className="text-sm font-semibold">{item.title}</p>
              {item.description && <p className="mt-0.5 break-words text-xs opacity-80">{item.description}</p>}
            </div>
            <button onClick={() => dismiss(item.id)} className="rounded p-0.5 opacity-60 hover:opacity-100">
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
        );
      })}
    </div>
  );
}
