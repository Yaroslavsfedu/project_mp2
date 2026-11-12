import { NavLink, Outlet, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import {
  Cpu,
  FileSearch,
  LayoutDashboard,
  LogOut,
  ScanSearch,
  Settings2,
  Sparkles,
  Upload,
} from "lucide-react";
import { systemApi } from "@/api/endpoints";
import { useAuth } from "@/store/auth";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui";

type NavItem = { to: string; label: string; icon: typeof LayoutDashboard; end?: boolean };

const NAV: NavItem[] = [
  { to: "/", label: "Рабочий стол", icon: LayoutDashboard, end: true },
  { to: "/documents", label: "Документы", icon: FileSearch },
  { to: "/upload", label: "Загрузка", icon: Upload },
  { to: "/search", label: "Умный поиск", icon: ScanSearch },
];

const ADMIN_NAV: NavItem = { to: "/admin", label: "Администрирование", icon: Settings2 };

function BrandMark() {
  return (
    <div className="flex items-center gap-2.5">
      <div className="grid h-9 w-9 place-items-center rounded-xl bg-gradient-to-br from-brand-500 to-brand-700 shadow-glow">
        <span className="font-mono text-sm font-bold text-white">IDP</span>
      </div>
      <div className="leading-tight">
        <p className="text-sm font-semibold text-white">ИКТИБ · IDP</p>
        <p className="text-[11px] text-slate-400">Локальная обработка документов</p>
      </div>
    </div>
  );
}

export function AppLayout() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const { data: status } = useQuery({
    queryKey: ["system-status"],
    queryFn: systemApi.status,
    refetchInterval: 60_000,
    retry: false,
  });

  const llmOnline = status?.llm.available;
  const ocrEngine = status?.ocr?.engine ?? "none";
  const ocrOnline = ocrEngine !== "none";
  const navItems = user?.role === "admin" ? [...NAV, ADMIN_NAV] : NAV;

  return (
    <div className="flex min-h-screen">
      <aside className="sticky top-0 hidden h-screen w-[248px] shrink-0 flex-col border-r border-white/[0.06] bg-ink-900/70 px-4 py-5 backdrop-blur lg:flex">
        <BrandMark />

        <nav className="mt-8 flex flex-col gap-1">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) => cn("nav-item", isActive && "nav-item-active")}
            >
              <item.icon className="h-[18px] w-[18px]" />
              {item.label}
            </NavLink>
          ))}
        </nav>

        <div className="mt-6 rounded-xl border border-white/[0.06] bg-white/[0.02] p-3">
          <p className="mb-2 flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-slate-400">
            <Cpu className="h-3.5 w-3.5" /> Локальный ИИ-стек
          </p>
          <div className="space-y-1.5 text-xs">
            <div className="flex items-center justify-between gap-2">
              <span className="text-slate-400">LLM</span>
              <Badge tone={llmOnline ? "brand" : "muted"} className="text-[10px]">
                <span
                  className={cn("h-1.5 w-1.5 rounded-full", llmOnline ? "bg-emerald-400" : "bg-amber-400")}
                />
                {llmOnline ? "Ollama" : "fallback"}
              </Badge>
            </div>
            <div className="flex items-center justify-between gap-2">
              <span className="text-slate-400">OCR</span>
              <Badge tone={ocrOnline ? "brand" : "muted"} className="text-[10px]">
                {ocrOnline ? ocrEngine : "без OCR"}
              </Badge>
            </div>
            <div className="flex items-center justify-between gap-2">
              <span className="text-slate-400">GPU</span>
              <Badge tone={status?.gpu?.available ? "brand" : "muted"} className="text-[10px]">
                {status?.gpu?.available
                  ? `${((status.gpu.memory_mb ?? 0) / 1024).toFixed(0)} ГБ`
                  : "CPU"}
              </Badge>
            </div>
            <div className="flex items-center justify-between gap-2">
              <span className="text-slate-400">Хранилище</span>
              <Badge tone="muted" className="text-[10px]">
                {status?.storage_backend ?? "local"}
              </Badge>
            </div>
          </div>
          <p className="mt-2.5 text-[10px] leading-relaxed text-slate-500">
            Все вычисления выполняются в локальном контуре, данные не покидают периметр.
          </p>
        </div>

        <div className="mt-auto space-y-3 pt-6">
          <div className="flex items-center gap-2.5 rounded-xl border border-white/[0.06] bg-white/[0.02] px-3 py-2.5">
            <div className="grid h-8 w-8 place-items-center rounded-lg bg-brand-600/20 text-xs font-semibold text-brand-200">
              {(user?.full_name || user?.email || "?").slice(0, 2).toUpperCase()}
            </div>
            <div className="min-w-0 flex-1 leading-tight">
              <p className="truncate text-xs font-medium text-slate-200">{user?.full_name || "Оператор"}</p>
              <p className="truncate text-[10px] text-slate-500">{user?.email}</p>
            </div>
            <button
              onClick={() => {
                logout();
                navigate("/login");
              }}
              className="rounded-lg p-1.5 text-slate-400 transition hover:bg-white/10 hover:text-white"
              title="Выйти"
            >
              <LogOut className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-30 flex items-center justify-between gap-4 border-b border-white/[0.06] bg-ink-950/80 px-4 py-3 backdrop-blur lg:hidden">
          <BrandMark />
          <div className="flex items-center gap-2">
            {navItems.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={({ isActive }) =>
                  cn("rounded-lg p-2 text-slate-400", isActive && "bg-brand-600/20 text-brand-200")
                }
              >
                <item.icon className="h-5 w-5" />
              </NavLink>
            ))}
          </div>
        </header>

        <main className="mx-auto w-full max-w-[1500px] flex-1 px-4 py-6 lg:px-8 lg:py-8">
          <Outlet />
        </main>

        <footer className="border-t border-white/[0.05] px-4 py-4 text-center text-[11px] text-slate-500 lg:px-8">
          Проектный офис ИКТИБ ЮФУ · IDP Platform v0.1 · On-Premise, без внешних API
          <Sparkles className="ml-1 inline h-3 w-3 text-brand-400" />
        </footer>
      </div>
    </div>
  );
}
