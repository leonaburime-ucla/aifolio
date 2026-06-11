import { create } from "zustand";
import type {
  AgUiWorkspaceStoreState,
  AgUiWorkspaceTab,
} from "@aifolio/contracts/entities/ag-ui";
export {
  resolveAgUiWorkspaceTab,
  toAgUiPageQuery,
} from "@aifolio/frontend-core/ag-ui";

/**
 * State for active tab selection in /ag-ui.
 */
export const useAgUiWorkspaceStore = create<AgUiWorkspaceStoreState>((set) => ({
  activeTab: "charts",
  setActiveTab: (tab) => set(() => ({ activeTab: tab })),
}));
export type { AgUiWorkspaceStoreState, AgUiWorkspaceTab };
