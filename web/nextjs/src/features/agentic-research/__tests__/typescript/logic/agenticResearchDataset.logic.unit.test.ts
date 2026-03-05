import { describe, expect, it, vi } from "vitest";
import { applyDatasetLoadReset } from "@/features/agentic-research/typescript/logic/agenticResearchDataset.logic";

describe("agenticResearchDataset.logic", () => {
  it("resets derived state slices before loading", () => {
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
