import { describe, expect, it, vi } from "vitest";
import {
  handleAgenticAddChartSpec,
  handleAgenticClearCharts,
  handleAgenticRemoveChartSpec,
  handleAgenticReorderChartSpecs,
} from "@/features/agentic-research/typescript/ai/tools/chartTools";

const { normalizeChartSpecInputMock } = vi.hoisted(() => ({
  normalizeChartSpecInputMock: vi.fn((input: unknown) => input),
}));

vi.mock("@/features/ag-ui-chat/typescript/logic/copilotAssistantPayload.util", () => ({
  normalizeChartSpecInput: normalizeChartSpecInputMock,
}));

const chart = (id: string) => ({
  id,
  title: id,
  type: "scatter" as const,
  xKey: "x",
  yKeys: ["y"],
  data: [],
});

describe("agentic chartTools", () => {
  it("adds normalized chart specs and returns ids", () => {
    const addChartSpec = vi.fn();

    expect(
      handleAgenticAddChartSpec({ chartSpec: chart("a") }, addChartSpec)
    ).toEqual({ status: "ok", addedCount: 1, ids: ["a"] });
    expect(
      handleAgenticAddChartSpec(
        { chartSpecs: [chart("a"), chart("b")] },
        addChartSpec
      )
    ).toEqual({ status: "ok", addedCount: 2, ids: ["a", "b"] });

    expect(addChartSpec).toHaveBeenCalledTimes(3);
  });

  it("returns INVALID_CHART_SPEC when payload cannot be normalized", () => {
    normalizeChartSpecInputMock.mockReturnValueOnce(null);
    const result = handleAgenticAddChartSpec({ chartSpec: null }, vi.fn());
    expect(result).toEqual({ status: "error", code: "INVALID_CHART_SPEC", addedCount: 0 });
  });

  it("clears charts and removes existing ids", () => {
    const clearFn = vi.fn();
    expect(handleAgenticClearCharts(clearFn)).toEqual({ status: "ok", cleared: true });
    expect(clearFn).toHaveBeenCalledTimes(1);

    const snapshot = { chartSpecs: [chart("a"), chart("b")] };
    const removeFn = vi.fn((id: string) => {
      snapshot.chartSpecs = snapshot.chartSpecs.filter((item) => item.id !== id);
    });

    expect(
      handleAgenticRemoveChartSpec("a", () => snapshot, removeFn)
    ).toEqual({ status: "ok", removed_chart_id: "a", remaining_count: 1 });

    expect(
      handleAgenticRemoveChartSpec("x", () => snapshot, removeFn)
    ).toEqual({
      status: "error",
      code: "CHART_NOT_FOUND",
      chart_id: "x",
      available_chart_ids: ["b"],
    });
  });

  it("reorders by ordered ids or index move and validates payload", () => {
    const snapshot = { chartSpecs: [chart("a"), chart("b"), chart("c")] };
    const reorderFn = vi.fn((orderedIds: string[]) => {
      const byId = new Map(snapshot.chartSpecs.map((item) => [item.id, item]));
      snapshot.chartSpecs = orderedIds.map((id) => byId.get(id)!).filter(Boolean);
    });

    expect(
      handleAgenticReorderChartSpecs(
        { ordered_ids: ["c", "a", "b"] },
        () => snapshot,
        reorderFn
      )
    ).toEqual({ status: "ok", mode: "ordered_ids", chart_ids: ["c", "a", "b"] });

    snapshot.chartSpecs = [chart("a"), chart("b"), chart("c")];
    expect(
      handleAgenticReorderChartSpecs(
        { from_index: 0, to_index: 2 },
        () => snapshot,
        reorderFn
      )
    ).toEqual({ status: "ok", mode: "index_move", chart_ids: ["b", "c", "a"] });

    expect(
      handleAgenticReorderChartSpecs(
        { from_index: 99, to_index: 0 },
        () => snapshot,
        reorderFn
      )
    ).toEqual({
      status: "error",
      code: "INDEX_OUT_OF_RANGE",
      from_index: 99,
      to_index: 0,
      chart_count: 3,
    });

    expect(
      handleAgenticReorderChartSpecs({}, () => snapshot, reorderFn)
    ).toEqual({
      status: "error",
      code: "INVALID_REORDER_PAYLOAD",
      hint: "Provide ordered_ids or both from_index and to_index.",
    });
  });
});
