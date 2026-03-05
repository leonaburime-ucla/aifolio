import { describe, expect, it, vi } from "vitest";
import { renderHook } from "@testing-library/react";

const {
  addChartSpec,
  clearChartSpecs,
  removeChartSpec,
  reorderChartSpecs,
  getState,
} = vi.hoisted(() => ({
  addChartSpec: vi.fn(),
  clearChartSpecs: vi.fn(),
  removeChartSpec: vi.fn(),
  reorderChartSpecs: vi.fn(),
  getState: vi.fn(() => ({ chartSpecs: [{ id: "x" }] })),
}));

vi.mock("@/features/agentic-research/typescript/react/state/zustand/agenticResearchChartStore", () => ({
  useAgenticResearchChartStore: Object.assign(
    (selector: (value: {
      addChartSpec: typeof addChartSpec;
      clearChartSpecs: typeof clearChartSpecs;
      removeChartSpec: typeof removeChartSpec;
      reorderChartSpecs: typeof reorderChartSpecs;
      chartSpecs: { id: string }[];
    }) => unknown) =>
      selector({
        addChartSpec,
        clearChartSpecs,
        removeChartSpec,
        reorderChartSpecs,
        chartSpecs: [{ id: "x" }],
      }),
    { getState }
  ),
}));

import {
  getAgenticResearchChartStateSnapshot,
  useAgenticResearchChartActionsAdapter,
} from "@/features/agentic-research/typescript/react/state/adapters/chartActions.adapter";

describe("chartActions.adapter", () => {
  it("exposes chart action functions and snapshot helper", () => {
    expect(getAgenticResearchChartStateSnapshot()).toEqual({ chartSpecs: [{ id: "x" }] });

    const { result } = renderHook(() => useAgenticResearchChartActionsAdapter());

    expect(result.current.chartSpecs).toEqual([{ id: "x" }]);
    result.current.addChartSpec({
      id: "a",
      title: "A",
      type: "scatter",
      xKey: "x",
      yKeys: ["y"],
      data: [],
    });
    result.current.clearChartSpecs();
    result.current.removeChartSpec("a");
    result.current.reorderChartSpecs(["a"]);

    expect(addChartSpec).toHaveBeenCalledTimes(1);
    expect(clearChartSpecs).toHaveBeenCalledTimes(1);
    expect(removeChartSpec).toHaveBeenCalledWith("a");
    expect(reorderChartSpecs).toHaveBeenCalledWith(["a"]);
    expect(result.current.getChartStateSnapshot()).toEqual({ chartSpecs: [{ id: "x" }] });
  });
});
