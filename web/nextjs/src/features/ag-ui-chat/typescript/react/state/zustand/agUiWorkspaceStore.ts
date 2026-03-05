import { create } from "zustand";
import type {
  AgUiWorkspaceStoreState,
  AgUiWorkspaceTab,
} from "@/features/ag-ui-chat/__types__/typescript/react/state/agUiWorkspace.types";
export {
  resolveAgUiWorkspaceTab,
  toAgUiPageQuery,
} from "@/features/ag-ui-chat/typescript/logic/agUiWorkspace.logic";

/**
 * State for active tab selection in /ag-ui.
 */
export const useAgUiWorkspaceStore = create<AgUiWorkspaceStoreState>((set) => ({
  activeTab: "charts",
  setActiveTab: (tab) => set(() => ({ activeTab: tab })),
}));
export type { AgUiWorkspaceStoreState, AgUiWorkspaceTab };
