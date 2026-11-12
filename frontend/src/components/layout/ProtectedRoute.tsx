import { useEffect } from "react";
import { Navigate, Outlet, useLocation } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { authApi } from "@/api/endpoints";
import { useAuth } from "@/store/auth";

/**
 * Защищённые маршруты. Сохранённый токен не считается входом, пока сервер его не подтвердил:
 * до ответа /auth/me интерфейс не показывается (иначе он мелькал и тут же уводил на /login).
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

  useEffect(() => {
    if (me.isError) useAuth.getState().logout();
  }, [me.isError]);

  if (!token || me.isError) {
    return <Navigate to="/login" replace state={{ from: location.pathname }} />;
  }
  if (me.isPending) {
    return (
      <div className="grid min-h-screen place-items-center text-sm text-slate-400">
        Проверка входа…
      </div>
    );
  }
  return <Outlet />;
}
