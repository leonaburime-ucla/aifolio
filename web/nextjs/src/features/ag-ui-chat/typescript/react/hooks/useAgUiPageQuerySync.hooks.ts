"use client";

import { useEffect } from "react";
import type { AgUiWorkspaceTab } from "@/features/ag-ui-chat/__types__/typescript/react/state/agUiWorkspace.types";

// ---------------------------------------------------------------------------
// Dependency Injection types
// ---------------------------------------------------------------------------

/**
 * Injected dependencies for the page-query-sync hook.
 */
export type AgUiPageQuerySyncDeps = {
  /** Current query-string search params (from Next.js). */
  searchParams: URLSearchParams;
  /** Current active workspace tab (from state adapter). */
  activeTab: AgUiWorkspaceTab;
  /** Sets the active workspace tab (from state adapter). */
  setActiveTab: (tab: AgUiWorkspaceTab) => void;
  /** Resolves a query page alias to a canonical tab. */
  resolveTab: (value: string) => AgUiWorkspaceTab | null;
  /** Determines next tab from query params vs current state. */
  resolveNextTabFromQuery: (params: {
    page: string | null;
    currentTab: AgUiWorkspaceTab;
    resolveTab: (value: string) => AgUiWorkspaceTab | null;
  }) => AgUiWorkspaceTab | null;
};

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

/**
 * Synchronizes `/ag-ui?page=<alias>` query params into AG-UI workspace state.
 *
 * All dependencies are injected via the `deps` bag — this hook does not import
 * adapters, stores, or logic directly.
 *
 * @param deps - Injected dependencies from orchestrator wiring.
 */
export function useAgUiPageQuerySync(deps: AgUiPageQuerySyncDeps): void {
  const { searchParams, activeTab, setActiveTab, resolveTab, resolveNextTabFromQuery } = deps;

  useEffect(() => {
    const page = searchParams.get("page");
    const resolved = resolveNextTabFromQuery({
      page,
      currentTab: activeTab,
      resolveTab,
    });
    if (!resolved) return;
    setActiveTab(resolved);
  }, [searchParams, activeTab, setActiveTab, resolveTab, resolveNextTabFromQuery]);
}
