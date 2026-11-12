import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { User } from "@/types";

/**
 * Источник истины для токена — память (стор). localStorage — только способ пережить перезагрузку.
 * В некоторых окружениях (iframe предпросмотра, блокировка хранилища) localStorage недоступен или не
 * сохраняет данные: тогда вход всё равно должен работать в текущей вкладке.
 */
const TOKEN_KEY = "idp-token";

export function readStoredToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

function writeStoredToken(token: string | null) {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* хранилище недоступно: работаем только с памятью */
  }
}

interface AuthState {
  token: string | null;
  user: User | null;
  setSession: (token: string, user: User) => void;
  setUser: (user: User) => void;
  logout: () => void;
}

export const useAuth = create<AuthState>()(
  persist(
    (set) => ({
      token: readStoredToken(),
      user: null,
      setSession: (token, user) => {
        writeStoredToken(token);
        set({ token, user });
      },
      setUser: (user) => set({ user }),
      logout: () => {
        writeStoredToken(null);
        set({ token: null, user: null });
      },
    }),
    {
      name: "idp-auth",
      partialize: (state) => ({ user: state.user }),
    },
  ),
);
