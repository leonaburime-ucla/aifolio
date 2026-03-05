import { describe, expect, it, vi } from "vitest";
import { renderHook } from "@testing-library/react";

const addChartSpec = vi.fn();
const clearChartSpecs = vi.fn();
const removeChartSpec = vi.fn();
const reorderChartSpecs = vi.fn();
const getChartStateSnapshot = vi.fn(() => ({ chartSpecs: [] }));
const setSelectedDatasetId = vi.fn();

vi.mock("@/features/agentic-research/typescript/react/state/adapters/chartActions.adapter", () => ({
  useAgenticResearchChartActionsAdapter: () => ({
    addChartSpec,
    clearChartSpecs,
    removeChartSpec,
    reorderChartSpecs,
    getChartStateSnapshot,
  }),
}));

vi.mock("@/features/agentic-research/typescript/react/state/adapters/agenticResearchState.adapter", () => ({
  useAgenticResearchStateAdapter: () => ({
    state: {
      datasetManifest: [{ id: "iris", label: "Iris" }],
    },
    actions: {
      setSelectedDatasetId,
    },
  }),
}));

vi.mock("@/features/agentic-research/typescript/ai/tools/chartTools", () => ({
  handleAgenticAddChartSpec: vi.fn(() => ({ status: "ok", addedCount: 1, ids: ["c1"] })),
  handleAgenticClearCharts: vi.fn(() => ({ status: "ok", cleared: true })),
  handleAgenticRemoveChartSpec: vi.fn(() => ({ status: "ok", removed_chart_id: "c1", remaining_count: 0 })),
  handleAgenticReorderChartSpecs: vi.fn(() => ({ status: "ok", mode: "ordered_ids", chart_ids: ["c1"] })),
}));

vi.mock("@/features/agentic-research/typescript/ai/tools/datasetTools", () => ({
  handleAgenticSetActiveDataset: vi.fn(() => ({ status: "ok", active_dataset_id: "iris" })),
}));

import { useAgenticResearchAiSurface } from "@/features/agentic-research/typescript/react/ai/adapters/useAgenticResearchAiSurface";
import {
  handleAgenticAddChartSpec,
  handleAgenticClearCharts,
  handleAgenticRemoveChartSpec,
  handleAgenticReorderChartSpecs,
} from "@/features/agentic-research/typescript/ai/tools/chartTools";
import { handleAgenticSetActiveDataset } from "@/features/agentic-research/typescript/ai/tools/datasetTools";

describe("useAgenticResearchAiSurface", () => {
  it("wires tool handlers to chart/state action ports", () => {
    const { result } = renderHook(() => useAgenticResearchAiSurface());

    expect(result.current.datasetManifest).toEqual([{ id: "iris", label: "Iris" }]);

    result.current.handlers.addChartSpec({ chartSpec: { id: "c1" } });
    result.current.handlers.clearCharts();
    result.current.handlers.removeChartSpec("c1");
    result.current.handlers.reorderChartSpecs({ ordered_ids: ["c1"] });
    result.current.handlers.setActiveDataset("iris");

    expect(handleAgenticAddChartSpec).toHaveBeenCalledWith(
      { chartSpec: { id: "c1" } },
      addChartSpec
    );
    expect(handleAgenticClearCharts).toHaveBeenCalledWith(clearChartSpecs);
    expect(handleAgenticRemoveChartSpec).toHaveBeenCalledWith(
      "c1",
      getChartStateSnapshot,
      removeChartSpec
    );
    expect(handleAgenticReorderChartSpecs).toHaveBeenCalledWith(
      { ordered_ids: ["c1"] },
      getChartStateSnapshot,
      reorderChartSpecs
    );
    expect(handleAgenticSetActiveDataset).toHaveBeenCalledWith(
      "iris",
      [{ id: "iris", label: "Iris" }],
      clearChartSpecs,
      setSelectedDatasetId
    );
  });
});
