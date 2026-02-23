import { beforeEach, describe, expect, it } from "vitest";
import { useAgenticResearchChartStore } from "@/features/agentic-research/state/zustand/agenticResearchChartStore";

describe("agenticResearchChartStore", () => {
  beforeEach(() => {
    useAgenticResearchChartStore.setState({ chartSpecs: [] });
  });

  it("deduplicates by id and prepends latest chart on add", () => {
    const chartA = { id: "a", title: "A", type: "line", data: [], xKey: "x", yKeys: ["y"] } as const;
    const chartB = { id: "b", title: "B", type: "line", data: [], xKey: "x", yKeys: ["y"] } as const;
    const chartB2 = { ...chartB, title: "B2" };

    useAgenticResearchChartStore.getState().addChartSpec(chartA as any);
    useAgenticResearchChartStore.getState().addChartSpec(chartB as any);
    useAgenticResearchChartStore.getState().addChartSpec(chartB2 as any);

    const ids = useAgenticResearchChartStore.getState().chartSpecs.map((spec) => spec.id);
    expect(ids).toEqual(["b", "a"]);
    expect(useAgenticResearchChartStore.getState().chartSpecs[0]?.title).toBe("B2");
  });

  it("removeChartSpec removes by exact id and clearChartSpecs resets to empty", () => {
    useAgenticResearchChartStore.getState().addChartSpec({
      id: "a",
      title: "A",
      type: "line",
      data: [],
      xKey: "x",
      yKeys: ["y"],
    } as any);
    useAgenticResearchChartStore.getState().addChartSpec({
      id: "b",
      title: "B",
      type: "line",
      data: [],
      xKey: "x",
      yKeys: ["y"],
    } as any);

    useAgenticResearchChartStore.getState().removeChartSpec("a");
    expect(useAgenticResearchChartStore.getState().chartSpecs.map((spec) => spec.id)).toEqual([
      "b",
    ]);

    useAgenticResearchChartStore.getState().clearChartSpecs();
    expect(useAgenticResearchChartStore.getState().chartSpecs).toEqual([]);
  });

  it("reorderChartSpecs reorders known ids and appends unspecified ids", () => {
    useAgenticResearchChartStore.getState().addChartSpec({
      id: "a",
      title: "A",
      type: "line",
      data: [],
      xKey: "x",
      yKeys: ["y"],
    } as any);
    useAgenticResearchChartStore.getState().addChartSpec({
      id: "b",
      title: "B",
      type: "line",
      data: [],
      xKey: "x",
      yKeys: ["y"],
    } as any);
    useAgenticResearchChartStore.getState().addChartSpec({
      id: "c",
      title: "C",
      type: "line",
      data: [],
      xKey: "x",
      yKeys: ["y"],
    } as any);
    // Current order is c, b, a due prepend behavior.
    useAgenticResearchChartStore.getState().reorderChartSpecs(["a", "c"]);
    expect(useAgenticResearchChartStore.getState().chartSpecs.map((spec) => spec.id)).toEqual([
      "a",
      "c",
      "b",
    ]);
  });
});
