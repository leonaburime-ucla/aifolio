import { describe, expect, it } from "vitest";
import { addChartSpecDedupPrepend } from "@aifolio/frontend-core/agentic-research";
import type { ChartSpec } from "@aifolio/contracts/entities/chart";

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

describe("REQ-005 chart add dedupe + prepend", () => {
  it("adding existing id [A,B] with B results in [B,A]", () => {
    const a = chart("A");
    const b = chart("B");

    const next = addChartSpecDedupPrepend({
      chartSpecs: [a, b],
      spec: b,
    });

    expect(next.map((item) => item.id)).toEqual(["B", "A"]);
  });
});
