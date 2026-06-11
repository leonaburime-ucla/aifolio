"use client";

import { useSearchParams } from "next/navigation";
import { useAgUiWorkspaceStateAdapter } from "@/features/ag-ui-chat/react/state/adapters/agUiWorkspaceState.adapter";
import {
  resolveAgUiWorkspaceTab,
  resolveNextAgUiTabFromQuery,
} from "@aifolio/frontend-core/ag-ui";
import { useAgUiPageQuerySync } from "@/features/ag-ui-chat/react/hooks/useAgUiPageQuerySync.hooks";

/**
 * Orchestrator that wires concrete dependencies for page-query-sync behavior.
 *
 * Imports adapters, logic, and Next.js navigation — the hook itself has
 * zero direct imports per Orc-BASH convention.
 */
export function useAgUiPageQuerySyncOrchestrator(): void {
  const searchParams = useSearchParams();
  const { activeTab, setActiveTab } = useAgUiWorkspaceStateAdapter();

  useAgUiPageQuerySync({
    searchParams,
    activeTab,
    setActiveTab,
    resolveTab: resolveAgUiWorkspaceTab,
    resolveNextTabFromQuery: resolveNextAgUiTabFromQuery,
  });
}
