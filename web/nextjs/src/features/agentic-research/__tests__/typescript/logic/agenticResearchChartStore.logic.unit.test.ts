import { describe, expect, it } from "vitest";
import {
  addChartSpecDedupPrepend,
  reorderChartSpecsWithRemainder,
} from "@/features/agentic-research/typescript/logic/agenticResearchChartStore.logic";

const chart = (id: string) => ({
  id,
  title: id,
  type: "scatter" as const,
  xKey: "x",
  yKeys: ["y"],
  data: [],
});

describe("agenticResearchChartStore.logic", () => {
  it("dedupes by id and prepends newest chart", () => {
    expect(
      addChartSpecDedupPrepend({
        chartSpecs: [chart("a"), chart("b")],
        spec: chart("b"),
      }).map((item) => item.id)
    ).toEqual(["b", "a"]);
  });

  it("reorders by id then appends unspecified ids in current order", () => {
    expect(
      reorderChartSpecsWithRemainder({
        chartSpecs: [chart("a"), chart("b"), chart("c")],
        orderedIds: ["c", "x", "a"],
      }).map((item) => item.id)
    ).toEqual(["c", "a", "b"]);
  });
});
