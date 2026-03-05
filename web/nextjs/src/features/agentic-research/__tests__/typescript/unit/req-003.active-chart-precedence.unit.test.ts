import { describe, expect, it } from "vitest";
import { resolveActiveChartSpec } from "@/features/agentic-research/typescript/logic/agenticResearchChart.logic";
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

describe("REQ-003 active chart precedence", () => {
  it("resolves pcaChartSpec ?? chartSpecs[0] ?? null", () => {
    const x = chart("x");
    const y = chart("y");
    const p = chart("p");

    expect(
      resolveActiveChartSpec({
        pcaChartSpec: null,
        chartSpecs: [x, y],
      })
    ).toEqual(x);

    expect(
      resolveActiveChartSpec({
        pcaChartSpec: p,
        chartSpecs: [x, y],
      })
    ).toEqual(p);

    expect(
      resolveActiveChartSpec({
        pcaChartSpec: null,
        chartSpecs: [],
      })
    ).toBeNull();
  });
});
