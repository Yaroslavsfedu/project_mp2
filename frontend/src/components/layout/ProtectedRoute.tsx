import { useEffect } from "react";
import { Navigate, Outlet, useLocation } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import axios from "axios";
import { apiError } from "@/api/client";
import { authApi } from "@/api/endpoints";
import { endSession, useAuth } from "@/store/auth";

function isUnauthorized(error: unknown): boolean {
  return axios.isAxiosError(error) && error.response?.status === 401;
}

/**
 * Защищённые маршруты. Сохранённый токен не считается входом, пока сервер его не подтвердил.
 * - 401 от /auth/me: токен не подходит -> сессия сбрасывается, открывается вход.
 * - Любая другая ошибка (сеть, 5xx, таймаут): сессия НЕ сбрасывается, показываем причину и кнопки.
 */
export function ProtectedRoute() {
  const token = useAuth((state) => state.token);
  const location = useLocation();

  const me = useQuery({
    queryKey: ["auth-me", token],
    queryFn: authApi.me,
    enabled: !!token,
    retry: false,
    staleTime: 60_000,
  });

  useEffect(() => {
    if (me.data) useAuth.getState().setUser(me.data);
  }, [me.data]);

  const unauthorized = me.isError && isUnauthorized(me.error);

  useEffect(() => {
    if (unauthorized) endSession(`Сервер отклонил сохранённую сессию (401): ${apiError(me.error)}. Войдите снова.`);
  }, [unauthorized, me.error]);

  if (!token || unauthorized) {
    return <Navigate to="/login" replace state={{ from: location.pathname }} />;
  }

  if (me.isPending) {
    return (
      <div className="grid min-h-screen place-items-center text-sm text-slate-400">Проверка входа…</div>
    );
  }

  if (me.isError) {
    return (
      <div className="grid min-h-screen place-items-center p-6">
        <div className="max-w-md space-y-4 rounded-xl border border-white/10 bg-ink-900/70 p-6 text-center">
          <p className="text-base font-semibold text-white">Не удалось проверить вход</p>
          <p className="text-sm text-slate-400">{apiError(me.error)}</p>
          <p className="text-xs text-slate-500">
            Сессия сохранена. Попробуйте ещё раз или войдите заново.
          </p>
          <div className="flex justify-center gap-3">
            <button className="btn-primary btn-sm" onClick={() => me.refetch()} disabled={me.isFetching}>
              {me.isFetching ? "Проверка…" : "Повторить"}
            </button>
            <button className="btn-ghost btn-sm" onClick={() => useAuth.getState().logout()}>
              Войти заново
            </button>
          </div>
        </div>
      </div>
    );
  }

  return <Outlet />;
}
