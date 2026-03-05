import { describe, expect, it, vi } from "vitest";
import { applyDatasetLoadReset } from "@/features/agentic-research/typescript/logic/agenticResearchDataset.logic";

describe("REQ-002 dataset reset before payload apply", () => {
  it("clears stale table/chart-derived state", () => {
    const actions = {
      setTableRows: vi.fn(),
      setTableColumns: vi.fn(),
      setNumericMatrix: vi.fn(),
      setFeatureNames: vi.fn(),
      setPcaChartSpec: vi.fn(),
    };

    applyDatasetLoadReset({ actions });

    expect(actions.setTableRows).toHaveBeenCalledWith([]);
    expect(actions.setTableColumns).toHaveBeenCalledWith([]);
    expect(actions.setNumericMatrix).toHaveBeenCalledWith([]);
    expect(actions.setFeatureNames).toHaveBeenCalledWith([]);
    expect(actions.setPcaChartSpec).toHaveBeenCalledWith(null);
  });
});
