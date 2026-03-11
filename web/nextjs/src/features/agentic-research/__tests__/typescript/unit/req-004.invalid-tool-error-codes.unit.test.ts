import { describe, expect, it, vi } from "vitest";
import {
  handleAgenticRemoveChartSpec,
  handleAgenticReorderChartSpecs,
} from "@/features/agentic-research/typescript/ai/tools/chartTools";
import { handleAgenticSetActiveDataset } from "@/features/agentic-research/typescript/ai/tools/datasetTools";
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

describe("REQ-004 stable structured tool error codes", () => {
  it("returns CHART_NOT_FOUND for unknown remove id", () => {
    const result = handleAgenticRemoveChartSpec(
      "missing",
      () => ({ chartSpecs: [chart("A"), chart("B")] }),
      vi.fn()
    );

    expect(result).toMatchObject({
      status: "error",
      code: "CHART_NOT_FOUND",
    });
  });

  it("returns INVALID_REORDER_PAYLOAD for malformed reorder args", () => {
    const result = handleAgenticReorderChartSpecs({}, () => ({ chartSpecs: [chart("A")] }), vi.fn());

    expect(result).toMatchObject({
      status: "error",
      code: "INVALID_REORDER_PAYLOAD",
    });
  });

  it("returns INVALID_DATASET_ID for unknown dataset token", () => {
    const result = handleAgenticSetActiveDataset(
      "unknown",
      [{ id: "wine", label: "Wine" }],
      vi.fn()
    );

    expect(result).toMatchObject({
      status: "error",
      code: "INVALID_DATASET_ID",
    });
  });
});
