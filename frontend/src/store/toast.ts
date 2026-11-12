import { create } from "zustand";

export type ToastKind = "success" | "error" | "info";

export interface Toast {
  id: number;
  kind: ToastKind;
  title: string;
  description?: string;
}

interface ToastState {
  toasts: Toast[];
  push: (kind: ToastKind, title: string, description?: string) => void;
  dismiss: (id: number) => void;
}

let counter = 0;

export const useToastStore = create<ToastState>((set) => ({
  toasts: [],
  push: (kind, title, description) => {
    const id = ++counter;
    set((state) => ({ toasts: [...state.toasts, { id, kind, title, description }] }));
    setTimeout(() => {
      set((state) => ({ toasts: state.toasts.filter((toast) => toast.id !== id) }));
    }, 5000);
  },
  dismiss: (id) => set((state) => ({ toasts: state.toasts.filter((toast) => toast.id !== id) })),
}));

export const toast = {
  success: (title: string, description?: string) => useToastStore.getState().push("success", title, description),
  error: (title: string, description?: string) => useToastStore.getState().push("error", title, description),
  info: (title: string, description?: string) => useToastStore.getState().push("info", title, description),
};
