import { describe, expect, it } from "vitest";
import { useAgenticResearchChartStore } from "@/features/agentic-research/typescript/react/state/zustand/agenticResearchChartStore";

const chart = (id: string) => ({
  id,
  title: id,
  type: "scatter" as const,
  xKey: "x",
  yKeys: ["y"],
  data: [],
});

describe("agenticResearchChartStore", () => {
  it("adds/removes/reorders/clears chart specs", () => {
    useAgenticResearchChartStore.setState({ chartSpecs: [] });

    const actions = useAgenticResearchChartStore.getState();
    actions.addChartSpec(chart("a"));
    actions.addChartSpec(chart("b"));
    actions.addChartSpec(chart("a"));
    expect(useAgenticResearchChartStore.getState().chartSpecs.map((c) => c.id)).toEqual([
      "a",
      "b",
    ]);

    actions.reorderChartSpecs(["b", "a"]);
    expect(useAgenticResearchChartStore.getState().chartSpecs.map((c) => c.id)).toEqual([
      "b",
      "a",
    ]);

    actions.removeChartSpec("b");
    expect(useAgenticResearchChartStore.getState().chartSpecs.map((c) => c.id)).toEqual([
      "a",
    ]);

    actions.clearChartSpecs();
    expect(useAgenticResearchChartStore.getState().chartSpecs).toEqual([]);
  });
});
