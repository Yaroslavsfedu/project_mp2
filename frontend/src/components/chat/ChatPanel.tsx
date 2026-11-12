import { useRef, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { Bot, CornerDownLeft, User } from "lucide-react";
import { apiError } from "@/api/client";
import { documentsApi } from "@/api/endpoints";
import { cn } from "@/lib/utils";
import { Spinner } from "@/components/ui";

interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: { page_number: number; snippet: string }[];
  engine?: string;
}

const SUGGESTIONS = [
  "Какая сумма договора?",
  "Кто подписал документ?",
  "Какие сроки поставки или оплаты?",
  "Перечисли ключевые реквизиты",
];

export function ChatPanel({ documentId }: { documentId: string }) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);

  const ask = useMutation({
    mutationFn: (text: string) =>
      documentsApi.chat(
        documentId,
        text,
        messages.map((message) => ({ role: message.role, content: message.content })),
      ),
    onSuccess: (data) => {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.answer, sources: data.sources, engine: data.engine },
      ]);
      requestAnimationFrame(() => scrollRef.current?.scrollTo({ top: 999999, behavior: "smooth" }));
    },
    onError: (error) => {
      setMessages((prev) => [...prev, { role: "assistant", content: `Ошибка: ${apiError(error)}` }]);
    },
  });

  const send = (text: string) => {
    const value = text.trim();
    if (!value || ask.isPending) return;
    setMessages((prev) => [...prev, { role: "user", content: value }]);
    setQuestion("");
    ask.mutate(value);
    requestAnimationFrame(() => scrollRef.current?.scrollTo({ top: 999999, behavior: "smooth" }));
  };

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex items-center justify-between border-b border-white/[0.06] px-4 py-2.5">
        <div>
          <p className="text-sm font-semibold text-white">Вопросы по документу (RAG)</p>
          <p className="text-[11px] text-slate-400">
            Ответ формируется локальной моделью по фрагментам этого документа
          </p>
        </div>
        <span className="chip border-white/10 bg-white/[0.03] text-[10px] text-slate-400">Ollama / fallback</span>
      </div>

      <div ref={scrollRef} className="flex-1 space-y-4 overflow-y-auto px-4 py-4">
        {messages.length === 0 && (
          <div className="rounded-xl border border-white/[0.07] bg-white/[0.02] p-4">
            <p className="flex items-center gap-2 text-sm font-medium text-slate-200">
              <Bot className="h-4 w-4 text-brand-300" /> Чем помочь по этому документу?
            </p>
            <div className="mt-3 flex flex-wrap gap-1.5">
              {SUGGESTIONS.map((suggestion) => (
                <button
                  key={suggestion}
                  onClick={() => send(suggestion)}
                  className="rounded-lg border border-white/10 bg-white/[0.03] px-2.5 py-1.5 text-xs text-slate-300 transition hover:border-brand-500/40 hover:text-white"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((message, index) => (
          <div key={index} className={cn("flex gap-2.5", message.role === "user" && "justify-end")}>
            {message.role === "assistant" && (
              <div className="mt-0.5 grid h-7 w-7 shrink-0 place-items-center rounded-lg bg-brand-600/20 text-brand-200">
                <Bot className="h-4 w-4" />
              </div>
            )}
            <div
              className={cn(
                "max-w-[85%] rounded-2xl px-3.5 py-2.5 text-sm leading-relaxed",
                message.role === "user"
                  ? "bg-brand-600/25 text-white"
                  : "border border-white/[0.07] bg-white/[0.03] text-slate-200",
              )}
            >
              <p className="whitespace-pre-wrap">{message.content}</p>
              {message.sources && message.sources.length > 0 && (
                <div className="mt-2 space-y-1 border-t border-white/10 pt-2">
                  <p className="text-[10px] uppercase tracking-wide text-slate-500">Источники</p>
                  {message.sources.map((source, i) => (
                    <p key={i} className="line-clamp-2 text-[11px] text-slate-400">
                      <span className="text-brand-300">стр. {source.page_number}:</span> {source.snippet}
                    </p>
                  ))}
                </div>
              )}
            </div>
            {message.role === "user" && (
              <div className="mt-0.5 grid h-7 w-7 shrink-0 place-items-center rounded-lg bg-white/10 text-slate-300">
                <User className="h-4 w-4" />
              </div>
            )}
          </div>
        ))}

        {ask.isPending && (
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <Spinner className="h-3.5 w-3.5" /> Модель анализирует документ…
          </div>
        )}
      </div>

      <div className="border-t border-white/[0.06] p-3">
        <div className="flex items-end gap-2">
          <textarea
            className="input min-h-[44px] resize-none"
            rows={1}
            placeholder="Задайте вопрос по содержанию документа…"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                send(question);
              }
            }}
          />
          <button className="btn-primary !px-3 !py-2.5" onClick={() => send(question)} disabled={ask.isPending}>
            <CornerDownLeft className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
