"use client";

import { useCallback, useMemo } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useAgUiWorkspaceStateAdapter } from "@/features/ag-ui-chat/typescript/react/state/adapters/agUiWorkspaceState.adapter";
import {
  AG_UI_WORKSPACE_TABS,
  buildAgUiWorkspaceTabHref,
} from "@/features/ag-ui-chat/typescript/logic/agUiWorkspace.logic";
import {
  useAgUiWorkspace,
  type AgUiWorkspaceResult,
} from "@/features/ag-ui-chat/typescript/react/hooks/useAgUiWorkspace.hooks";
import type { AgUiWorkspaceTab } from "@/features/ag-ui-chat/__types__/typescript/react/state/agUiWorkspace.types";

/**
 * Orchestrator that wires concrete dependencies for workspace tab behavior.
 *
 * Imports adapters, logic, and Next.js navigation — the hook itself has
 * zero direct imports per Orc-BASH convention.
 *
 * @returns Active tab, tab definitions, and click handler for tab navigation.
 */
export function useAgUiWorkspaceOrchestrator(): AgUiWorkspaceResult {
  const { activeTab, setActiveTab } = useAgUiWorkspaceStateAdapter();
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  const buildTabHref = useCallback(
    (tab: AgUiWorkspaceTab) =>
      buildAgUiWorkspaceTabHref({
        pathname,
        searchParams: new URLSearchParams(searchParams.toString()),
        tab,
      }),
    [pathname, searchParams]
  );

  const replaceUrl = useCallback(
    (href: string) => router.replace(href),
    [router]
  );

  const deps = useMemo(
    () => ({
      activeTab,
      setActiveTab,
      tabs: AG_UI_WORKSPACE_TABS,
      buildTabHref,
      replaceUrl,
    }),
    [activeTab, setActiveTab, buildTabHref, replaceUrl]
  );

  return useAgUiWorkspace(deps);
}
