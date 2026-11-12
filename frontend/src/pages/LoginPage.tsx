import { useState } from "react";
import { Navigate, useLocation, useNavigate } from "react-router-dom";
import { useMutation } from "@tanstack/react-query";
import { ArrowRight, FileScan, Lock, Mail, ShieldCheck, Sparkles, Zap } from "lucide-react";
import { authApi } from "@/api/endpoints";
import { apiError } from "@/api/client";
import { useAuth } from "@/store/auth";
import { toast } from "@/store/toast";
import { ErrorBanner, Spinner } from "@/components/ui";

const FEATURES = [
  { icon: FileScan, title: "OCR с координатами", text: "PaddleOCR / Tesseract, Bounding Boxes и confidence по каждой строке" },
  { icon: Sparkles, title: "Локальная LLM", text: "Qwen 2.5 через Ollama: коррекция, NER, классификация, резюме" },
  { icon: Zap, title: "Гибридный поиск", text: "Полнотекстовый + векторный поиск с подсветкой совпадений" },
  { icon: ShieldCheck, title: "On-Premise", text: "Данные не покидают контур: 152-ФЗ, изолированная сеть Docker" },
];

export function LoginPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const { token, setSession } = useAuth();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("demo@idp-project.ru");
  const [password, setPassword] = useState("demo12345");
  const [fullName, setFullName] = useState("");
  const [error, setError] = useState<string | null>(null);

  const submit = useMutation({
    mutationFn: async () =>
      mode === "login"
        ? authApi.login({ email, password })
        : authApi.register({ email, password, full_name: fullName || "Оператор" }),
    onSuccess: (data) => {
      setSession(data.access_token, data.user);
      toast.success(`Добро пожаловать, ${data.user.full_name || data.user.email}`);
      const from = (location.state as { from?: string } | null)?.from;
      navigate(from && from !== "/login" ? from : "/", { replace: true });
    },
    onError: (err) => setError(apiError(err)),
  });

  if (token) return <Navigate to="/" replace />;

  return (
    <div className="grid min-h-screen lg:grid-cols-[1.15fr,1fr]">
      <div className="relative hidden flex-col justify-between overflow-hidden border-r border-white/[0.06] bg-ink-900/40 p-10 lg:flex">
        <div className="pointer-events-none absolute -left-24 top-10 h-80 w-80 rounded-full bg-brand-600/20 blur-3xl" />
        <div className="pointer-events-none absolute bottom-0 right-0 h-96 w-96 rounded-full bg-indigo-600/10 blur-3xl" />

        <div className="relative flex items-center gap-3">
          <div className="grid h-11 w-11 place-items-center rounded-xl bg-gradient-to-br from-brand-500 to-brand-700 shadow-glow">
            <span className="font-mono text-sm font-bold text-white">IDP</span>
          </div>
          <div>
            <p className="text-sm font-semibold text-white">Проектный офис ИКТИБ</p>
            <p className="text-xs text-slate-400">Институт компьютерных технологий и информационной безопасности ЮФУ</p>
          </div>
        </div>

        <div className="relative max-w-xl">
          <p className="mb-3 inline-flex items-center gap-2 rounded-full border border-brand-500/30 bg-brand-500/10 px-3 py-1 text-xs font-medium text-brand-200">
            <Sparkles className="h-3.5 w-3.5" /> Междисциплинарный проект · 2024/25
          </p>
          <h1 className="text-4xl font-bold leading-tight tracking-tight text-white">
            Оцифровка документов и&nbsp;интеллектуальная обработка информации
          </h1>
          <p className="mt-4 text-sm leading-relaxed text-slate-400">
            Единое окно для загрузки сканов и PDF, распознавания текста, извлечения сущностей, поиска по архиву
            и ручной верификации результатов — полностью в локальном контуре.
          </p>

          <div className="mt-8 grid gap-3 sm:grid-cols-2">
            {FEATURES.map((feature) => (
              <div key={feature.title} className="rounded-xl border border-white/[0.07] bg-white/[0.02] p-3.5">
                <feature.icon className="h-4 w-4 text-brand-300" />
                <p className="mt-2 text-sm font-medium text-slate-100">{feature.title}</p>
                <p className="mt-0.5 text-[11px] leading-relaxed text-slate-400">{feature.text}</p>
              </div>
            ))}
          </div>
        </div>

        <p className="relative text-[11px] text-slate-500">
          Наставник проекта: Семенов В. Е., м.н.с. НИИ МВС ЮФУ · vse@sfedu.ru
        </p>
      </div>

      <div className="flex items-center justify-center px-5 py-12">
        <div className="w-full max-w-sm">
          <div className="mb-8 flex items-center gap-2.5 lg:hidden">
            <div className="grid h-9 w-9 place-items-center rounded-xl bg-gradient-to-br from-brand-500 to-brand-700">
              <span className="font-mono text-xs font-bold text-white">IDP</span>
            </div>
            <p className="text-sm font-semibold text-white">ИКТИБ · IDP Платформа</p>
          </div>

          <h2 className="text-2xl font-semibold tracking-tight text-white">
            {mode === "login" ? "Вход в платформу" : "Регистрация"}
          </h2>
          <p className="mt-1 text-sm text-slate-400">
            {mode === "login"
              ? "Доступ к рабочему пространству оператора"
              : "Создайте учётную запись оператора верификации"}
          </p>

          <form
            className="mt-6 space-y-3.5"
            onSubmit={(event) => {
              event.preventDefault();
              setError(null);
              submit.mutate();
            }}
          >
            {mode === "register" && (
              <div>
                <label className="label">ФИО</label>
                <input
                  className="input"
                  value={fullName}
                  onChange={(event) => setFullName(event.target.value)}
                  placeholder="Иванов Иван Иванович"
                />
              </div>
            )}

            <div>
              <label className="label">Email</label>
              <div className="relative">
                <Mail className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
                <input
                  className="input pl-9"
                  type="email"
                  required
                  value={email}
                  onChange={(event) => setEmail(event.target.value)}
                  placeholder="operator@example.ru"
                />
              </div>
            </div>

            <div>
              <label className="label">Пароль</label>
              <div className="relative">
                <Lock className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
                <input
                  className="input pl-9"
                  type="password"
                  required
                  minLength={6}
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  placeholder="••••••••"
                />
              </div>
            </div>

            {error && <ErrorBanner message={error} />}

            <button type="submit" className="btn-primary w-full" disabled={submit.isPending}>
              {submit.isPending ? <Spinner /> : null}
              {mode === "login" ? "Войти" : "Создать аккаунт"}
              {!submit.isPending && <ArrowRight className="h-4 w-4" />}
            </button>
          </form>

          <div className="mt-4 flex items-center justify-between text-xs text-slate-400">
            <button
              className="underline-offset-2 hover:text-slate-200 hover:underline"
              onClick={() => setMode(mode === "login" ? "register" : "login")}
            >
              {mode === "login" ? "Нет аккаунта? Зарегистрироваться" : "Уже есть аккаунт? Войти"}
            </button>
          </div>

          <div className="mt-8 rounded-xl border border-white/[0.07] bg-white/[0.02] p-3.5">
            <p className="text-[11px] font-semibold uppercase tracking-wide text-slate-400">Демо-доступ</p>
            <p className="mt-1 text-xs text-slate-400">
              <span className="font-mono text-slate-200">demo@idp-project.ru</span> /{" "}
              <span className="font-mono text-slate-200">demo12345</span>
            </p>
            <button
              className="mt-2 text-xs font-medium text-brand-300 hover:text-brand-200"
              onClick={() => {
                setEmail("demo@idp-project.ru");
                setPassword("demo12345");
                setMode("login");
                setError(null);
                authApi
                  .login({ email: "demo@idp-project.ru", password: "demo12345" })
                  .then((data) => {
                    setSession(data.access_token, data.user);
                    navigate("/", { replace: true });
                  })
                  .catch((err) => setError(apiError(err)));
              }}
            >
              Войти как демо-оператор →
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
