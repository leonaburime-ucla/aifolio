import type { AgUiWorkspaceTab } from "@/features/ag-ui-chat/__types__/typescript/react/state/agUiWorkspace.types";

/**
 * AG-UI workspace domain logic.
 *
 * Purpose:
 * - Resolve user/tool tab aliases to canonical tab ids.
 * - Map canonical tabs to URL query aliases.
 * - Provide workspace tab definitions and URL/query state helpers.
 *
 * Layering:
 * - Pure logic module with no React/Zustand/runtime dependencies.
 */

const TAB_ALIASES: Record<string, AgUiWorkspaceTab> = {
  base: "charts",
  charts: "charts",
  chart: "charts",
  home: "charts",
  ar: "agentic-research",
  "agentic research": "agentic-research",
  "agentic-research": "agentic-research",
  research: "agentic-research",
  pca: "agentic-research",
  pytorch: "pytorch",
  torch: "pytorch",
  tensorflow: "tensorflow",
  tf: "tensorflow",
};

export const AG_UI_WORKSPACE_TABS: ReadonlyArray<{ id: AgUiWorkspaceTab; label: string }> = [
  { id: "charts", label: "Charts" },
  { id: "agentic-research", label: "Agentic Research" },
  { id: "pytorch", label: "PyTorch" },
  { id: "tensorflow", label: "Tensorflow" },
];

/**
 * Resolves flexible model/user input into a canonical AG-UI tab.
 */
export function resolveAgUiWorkspaceTab(value: string): AgUiWorkspaceTab | null {
  const normalized = String(value || "")
    .trim()
    .toLowerCase();

  if (!normalized) return null;
  if (normalized in TAB_ALIASES) return TAB_ALIASES[normalized];
  return null;
}

/**
 * Converts a canonical workspace tab into `/ag-ui?page=...` query alias.
 */
export function toAgUiPageQuery(tab: AgUiWorkspaceTab): "base" | "ar" | "pytorch" | "tensorflow" {
  if (tab === "charts") return "base";
  if (tab === "agentic-research") return "ar";
  return tab;
}

/**
 * Builds the URL for a target AG-UI tab while preserving existing query params.
 */
export function buildAgUiWorkspaceTabHref({
  pathname,
  searchParams,
  tab,
}: {
  pathname: string;
  searchParams: URLSearchParams;
  tab: AgUiWorkspaceTab;
}): string {
  const params = new URLSearchParams(searchParams.toString());
  params.set("page", toAgUiPageQuery(tab));
  return `${pathname}?${params.toString()}`;
}

/**
 * Resolves next AG-UI tab state from a query value and current tab.
 */
export function resolveNextAgUiTabFromQuery({
  page,
  currentTab,
  resolveTab,
}: {
  page: string | null;
  currentTab: AgUiWorkspaceTab;
  resolveTab: (value: string) => AgUiWorkspaceTab | null;
}): AgUiWorkspaceTab | null {
  if (!page) return null;
  const resolved = resolveTab(page);
  if (!resolved || resolved === currentTab) return null;
  return resolved;
}
