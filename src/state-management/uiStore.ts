import { create } from "zustand";

interface UIState {
  theme: "light" | "dark";
  isSidebarOpen: boolean;
  toggleSidebar: () => void;
  setTheme: (theme: "light" | "dark") => void;
}

export const createUIStore = () =>
  create<UIState>((set) => ({
    theme: "light",
    isSidebarOpen: true,
    toggleSidebar: () => set((state) => ({ isSidebarOpen: !state.isSidebarOpen })),
    setTheme: (theme) => set({ theme }),
  }));
