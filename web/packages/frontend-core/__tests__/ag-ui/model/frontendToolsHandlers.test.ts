import { describe, expect, it, vi } from "vitest";
import {
  handleAddChartSpec,
  handleNavigateToPage,
  handleSwitchAgUiTab,
} from "../../../src/ag-ui/model/frontendTools";

describe("handleAddChartSpec", () => {
  it("adds a single chart spec", () => {
    const addChartSpec = vi.fn();
    const result = handleAddChartSpec(
      { chartSpec: { id: "c1", type: "bar", xKey: "x", yKeys: ["y"], data: [{ x: 1, y: 2 }], title: "Test" } },
      addChartSpec
    );
    expect(result.status).toBe("ok");
    expect(result.addedCount).toBe(1);
    expect(addChartSpec).toHaveBeenCalledTimes(1);
  });

  it("adds multiple chart specs", () => {
    const addChartSpec = vi.fn();
    const result = handleAddChartSpec(
      {
        chartSpecs: [
          { id: "c1", type: "bar", xKey: "x", yKeys: ["y"], data: [{ x: 1, y: 2 }], title: "A" },
          { id: "c2", type: "line", xKey: "x", yKeys: ["y"], data: [{ x: 1, y: 3 }], title: "B" },
        ],
      },
      addChartSpec
    );
    expect(result.status).toBe("ok");
    expect(result.addedCount).toBe(2);
    expect(addChartSpec).toHaveBeenCalledTimes(2);
  });

  it("returns error for invalid chart spec input", () => {
    const addChartSpec = vi.fn();
    const result = handleAddChartSpec(
      { chartSpec: { invalid: true } as any },
      addChartSpec
    );
    expect(result.status).toBe("error");
    expect(result.addedCount).toBe(0);
    expect(addChartSpec).not.toHaveBeenCalled();
  });
});

describe("handleNavigateToPage", () => {
  it("resolves a valid route alias", () => {
    const result = handleNavigateToPage("chat");
    expect(result.status).toBe("ok");
    if (result.status === "ok") {
      expect(result.resolvedRoute).toBeTruthy();
    }
  });

  it("returns error for unknown route", () => {
    const result = handleNavigateToPage("nonexistent-page-xyz");
    expect(result.status).toBe("error");
  });
});

describe("handleSwitchAgUiTab", () => {
  it("resolves a valid tab name", () => {
    const result = handleSwitchAgUiTab("charts");
    expect(result.status).toBe("ok");
    if (result.status === "ok") {
      expect(result.tab).toBe("charts");
    }
  });

  it("returns error for invalid tab", () => {
    const result = handleSwitchAgUiTab("invalid-tab-xyz");
    expect(result.status).toBe("error");
  });
});
