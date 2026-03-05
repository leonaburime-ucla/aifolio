import { describe, expect, it } from "vitest";
import { reorderChartSpecsWithRemainder } from "@/features/agentic-research/typescript/logic/agenticResearchChartStore.logic";
import type { ChartSpec } from "@/features/ai-chat/__types__/typescript/chart.types";

function chart(id: string): ChartSpec {
  return {
    id,
    title: id,
    type: "scatter",
    xKey: "x",
    yKeys: ["y"],
    data: [],
  };
}

describe("DR-003 reorder appends unspecified ids", () => {
  it("keeps unspecified chart ids in current order after ordered ids", () => {
    const a = chart("A");
    const b = chart("B");
    const c = chart("C");

    const next = reorderChartSpecsWithRemainder({
      chartSpecs: [a, b, c],
      orderedIds: ["C"],
    });

    expect(next.map((item) => item.id)).toEqual(["C", "A", "B"]);
  });
});
