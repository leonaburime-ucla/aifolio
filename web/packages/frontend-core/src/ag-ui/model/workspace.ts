import type { AgUiWorkspaceTab } from "@aifolio/contracts/entities/ag-ui";

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

export function resolveAgUiWorkspaceTab(value: string): AgUiWorkspaceTab | null {
  const normalized = String(value || "")
    .trim()
    .toLowerCase();

  if (!normalized) return null;
  if (normalized in TAB_ALIASES) return TAB_ALIASES[normalized];
  return null;
}

export function toAgUiPageQuery(tab: AgUiWorkspaceTab): "base" | "ar" | "pytorch" | "tensorflow" {
  if (tab === "charts") return "base";
  if (tab === "agentic-research") return "ar";
  return tab;
}

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
