"use client";

import { useCallback } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import type { AgUiWorkspaceTab } from "@/features/ag-ui-chat/typescript/react/state/zustand/agUiWorkspaceStore";
import { useAgUiWorkspaceStateAdapter } from "@/features/ag-ui-chat/typescript/react/state/adapters/agUiWorkspaceState.adapter";
import {
  AG_UI_WORKSPACE_TABS,
  buildAgUiWorkspaceTabHref,
} from "@/features/ag-ui-chat/typescript/logic/agUiWorkspace.logic";

/**
 * Provides AG-UI workspace tab view model and tab-change behavior.
 *
 * @returns Active tab, tab definitions, and click handler for tab navigation.
 */
export function useAgUiWorkspace() {
  const { activeTab, setActiveTab } = useAgUiWorkspaceStateAdapter();
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  const handleTabClick = useCallback(
    (tab: AgUiWorkspaceTab) => {
      setActiveTab(tab);
      const href = buildAgUiWorkspaceTabHref({
        pathname,
        searchParams: new URLSearchParams(searchParams.toString()),
        tab,
      });
      router.replace(href);
    },
    [pathname, router, searchParams, setActiveTab]
  );

  return {
    activeTab,
    tabs: AG_UI_WORKSPACE_TABS,
    handleTabClick,
  };
}
