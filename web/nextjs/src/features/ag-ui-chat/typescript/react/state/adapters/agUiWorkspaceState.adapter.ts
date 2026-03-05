import { useAgUiWorkspaceStore } from "@/features/ag-ui-chat/typescript/react/state/zustand/agUiWorkspaceStore";
import type { AgUiWorkspaceStatePort } from "@/features/ag-ui-chat/__types__/typescript/react/state/agUiWorkspace.types";

/**
 * Adapter exposing AG-UI workspace tab state via a narrow state port.
 */
export function useAgUiWorkspaceStateAdapter(): AgUiWorkspaceStatePort {
  const activeTab = useAgUiWorkspaceStore((state) => state.activeTab);
  const setActiveTab = useAgUiWorkspaceStore((state) => state.setActiveTab);

  return {
    activeTab,
    setActiveTab,
  };
}
