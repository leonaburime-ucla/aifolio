import { describe, expect, it } from "vitest";
import {
  AG_UI_WORKSPACE_TABS,
  buildAgUiWorkspaceTabHref,
  resolveNextAgUiTabFromQuery,
} from "@/features/ag-ui-chat/typescript/logic/agUiWorkspace.logic";
import { resolveAgUiWorkspaceTab } from "@/features/ag-ui-chat/typescript/logic/agUiWorkspace.logic";

describe("AG_UI_WORKSPACE_TABS", () => {
  it("defines expected tabs", () => {
    expect(AG_UI_WORKSPACE_TABS.map((tab) => tab.id)).toEqual([
      "charts",
      "agentic-research",
      "pytorch",
      "tensorflow",
    ]);
  });
});

describe("buildAgUiWorkspaceTabHref", () => {
  it("sets page query while preserving existing params", () => {
    const href = buildAgUiWorkspaceTabHref({
      pathname: "/ag-ui",
      searchParams: new URLSearchParams("foo=bar&page=charts"),
      tab: "pytorch",
    });

    expect(href).toBe("/ag-ui?foo=bar&page=pytorch");
  });
});

describe("resolveNextAgUiTabFromQuery", () => {
  it("returns next tab when query resolves to a different tab", () => {
    const next = resolveNextAgUiTabFromQuery({
      page: "pytorch",
      currentTab: "charts",
      resolveTab: resolveAgUiWorkspaceTab,
    });
    expect(next).toBe("pytorch");
  });

  it("returns null when query tab is missing, invalid, or unchanged", () => {
    expect(
      resolveNextAgUiTabFromQuery({
        page: null,
        currentTab: "charts",
        resolveTab: resolveAgUiWorkspaceTab,
      })
    ).toBeNull();
    expect(
      resolveNextAgUiTabFromQuery({
        page: "unknown",
        currentTab: "charts",
        resolveTab: resolveAgUiWorkspaceTab,
      })
    ).toBeNull();
    expect(
      resolveNextAgUiTabFromQuery({
        page: "charts",
        currentTab: "charts",
        resolveTab: resolveAgUiWorkspaceTab,
      })
    ).toBeNull();
  });
});
