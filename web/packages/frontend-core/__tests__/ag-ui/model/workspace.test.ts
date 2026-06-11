import { describe, expect, it } from "vitest";
import {
  AG_UI_WORKSPACE_TABS,
  buildAgUiWorkspaceTabHref,
  resolveNextAgUiTabFromQuery,
  resolveAgUiWorkspaceTab,
  toAgUiPageQuery,
} from "@aifolio/frontend-core/ag-ui";

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

describe("resolveAgUiWorkspaceTab", () => {
  it("resolves aliases and rejects blank values", () => {
    expect(resolveAgUiWorkspaceTab(" chart ")).toBe("charts");
    expect(resolveAgUiWorkspaceTab("AR")).toBe("agentic-research");
    expect(resolveAgUiWorkspaceTab("pca")).toBe("agentic-research");
    expect(resolveAgUiWorkspaceTab("torch")).toBe("pytorch");
    expect(resolveAgUiWorkspaceTab("tf")).toBe("tensorflow");
    expect(resolveAgUiWorkspaceTab("   ")).toBeNull();
    expect(resolveAgUiWorkspaceTab(undefined as never)).toBeNull();
  });
});

describe("toAgUiPageQuery", () => {
  it("maps canonical tabs into URL page aliases", () => {
    expect(toAgUiPageQuery("charts")).toBe("base");
    expect(toAgUiPageQuery("agentic-research")).toBe("ar");
    expect(toAgUiPageQuery("pytorch")).toBe("pytorch");
    expect(toAgUiPageQuery("tensorflow")).toBe("tensorflow");
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
