"use client";

import { useCallback } from "react";
import type { AgUiWorkspaceTab } from "@/features/ag-ui-chat/__types__/typescript/react/state/agUiWorkspace.types";

// ---------------------------------------------------------------------------
// Dependency Injection types
// ---------------------------------------------------------------------------

/**
 * Tab definition entry for workspace tab list.
 */
export type AgUiWorkspaceTabDef = {
  id: AgUiWorkspaceTab;
  label: string;
};

/**
 * Injected dependencies for the workspace hook.
 */
export type AgUiWorkspaceDeps = {
  /** Current active workspace tab (from state adapter). */
  activeTab: AgUiWorkspaceTab;
  /** Sets the active workspace tab (from state adapter). */
  setActiveTab: (tab: AgUiWorkspaceTab) => void;
  /** Available tab definitions. */
  tabs: ReadonlyArray<AgUiWorkspaceTabDef>;
  /** Builds the URL href for a given tab (from logic). */
  buildTabHref: (tab: AgUiWorkspaceTab) => string;
  /** Replaces the current URL without full navigation (from router). */
  replaceUrl: (href: string) => void;
};

/**
 * Return type for the workspace hook.
 */
export type AgUiWorkspaceResult = {
  activeTab: AgUiWorkspaceTab;
  tabs: ReadonlyArray<AgUiWorkspaceTabDef>;
  handleTabClick: (tab: AgUiWorkspaceTab) => void;
};

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

/**
 * Provides AG-UI workspace tab view model and tab-change behavior.
 *
 * All dependencies are injected via the `deps` bag — this hook does not import
 * adapters, stores, or logic directly.
 *
 * @param deps - Injected dependencies from orchestrator wiring.
 * @returns Active tab, tab definitions, and click handler for tab navigation.
 */
export function useAgUiWorkspace(deps: AgUiWorkspaceDeps): AgUiWorkspaceResult {
  const { activeTab, setActiveTab, tabs, buildTabHref, replaceUrl } = deps;

  const handleTabClick = useCallback(
    (tab: AgUiWorkspaceTab) => {
      setActiveTab(tab);
      replaceUrl(buildTabHref(tab));
    },
    [setActiveTab, buildTabHref, replaceUrl]
  );

  return {
    activeTab,
    tabs,
    handleTabClick,
  };
}
