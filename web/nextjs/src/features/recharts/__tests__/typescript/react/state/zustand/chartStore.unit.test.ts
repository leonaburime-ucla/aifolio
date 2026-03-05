import { describe, expect, it } from "vitest";
import { useChartStore } from "@/features/recharts/typescript/react/state/zustand/chartStore";

const chart = (id: string) => ({
  id,
  title: id,
  type: "line" as const,
  xKey: "x",
  yKeys: ["y"],
  data: [],
});

describe("chartStore", () => {
  it("adds, dedupes, removes and clears chart specs", () => {
    useChartStore.setState({ chartSpecs: [] });
    const state = useChartStore.getState();

    state.addChartSpec(chart("a"));
    state.addChartSpec(chart("b"));
    state.addChartSpec(chart("a"));
    expect(useChartStore.getState().chartSpecs.map((item) => item.id)).toEqual([
      "a",
      "b",
    ]);

    state.removeChartSpec("a");
    expect(useChartStore.getState().chartSpecs.map((item) => item.id)).toEqual([
      "b",
    ]);

    state.clearChartSpecs();
    expect(useChartStore.getState().chartSpecs).toEqual([]);
  });
});
