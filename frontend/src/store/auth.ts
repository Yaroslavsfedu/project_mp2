import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { User } from "@/types";

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
      token: localStorage.getItem("idp-token"),
      user: null,
      setSession: (token, user) => {
        localStorage.setItem("idp-token", token);
        set({ token, user });
      },
      setUser: (user) => set({ user }),
      logout: () => {
        localStorage.removeItem("idp-token");
        set({ token: null, user: null });
      },
    }),
    {
      name: "idp-auth",
      partialize: (state) => ({ user: state.user }),
    },
  ),
);
