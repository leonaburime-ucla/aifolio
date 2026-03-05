"use client";

import { useEffect } from "react";
import { useSearchParams } from "next/navigation";
import { useAgUiWorkspaceStateAdapter } from "@/features/ag-ui-chat/typescript/react/state/adapters/agUiWorkspaceState.adapter";
import { useAgUiWorkspaceStore } from "@/features/ag-ui-chat/typescript/react/state/zustand/agUiWorkspaceStore";
import {
  resolveAgUiWorkspaceTab,
  resolveNextAgUiTabFromQuery,
} from "@/features/ag-ui-chat/typescript/logic/agUiWorkspace.logic";

/**
 * Synchronizes `/ag-ui?page=<alias>` query params into AG-UI workspace state.
 */
export function useAgUiPageQuerySync(): void {
  const searchParams = useSearchParams();
  const { setActiveTab } = useAgUiWorkspaceStateAdapter();

  useEffect(() => {
    const current = useAgUiWorkspaceStore.getState().activeTab;
    const page = searchParams.get("page");
    const resolved = resolveNextAgUiTabFromQuery({
      page,
      currentTab: current,
      resolveTab: resolveAgUiWorkspaceTab,
    });
    if (!resolved) return;
    setActiveTab(resolved);
    console.log("[agui-debug] workspace.query_tab_applied", { page, resolved });
  }, [searchParams, setActiveTab]);
}
