import { renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.mock("@/features/ml/typescript/react/orchestrators/mlDatasetOrchestrator", () => ({
  useMlDatasetOrchestrator: () => ({
    datasetOptions: [],
    selectedDatasetId: "d1",
    setSelectedDatasetId: vi.fn(),
    isLoading: false,
    error: null,
    tableRows: [],
    tableColumns: ["target"],
    rowCount: 0,
    totalRowCount: 0,
    reloadManifest: vi.fn(),
    reloadDataset: vi.fn(),
  }),
}));

import { useTensorflowTrainingIntegration } from "@/features/ml/typescript/react/hooks/useTensorflowTraining.hooks";

describe("useTensorflowTrainingIntegration", () => {
  it("combines dataset/ui/logic/training runs", () => {
    const { result } = renderHook(() =>
      useTensorflowTrainingIntegration({
        useTrainingRunsState: () => ({
          trainingRuns: [{ run_id: "r1" }],
          prependTrainingRun: vi.fn(),
          clearTrainingRuns: vi.fn(),
        }),
        runTraining: vi.fn(async () => ({
          stopped: false,
          completed: 0,
          total: 0,
          completedTeacherRuns: [],
          failedRuns: 0,
          firstFailureMessage: null,
        })),
        runDistillation: vi.fn(async () => ({ status: "error", error: "x" })),
      })
    );

    expect(result.current.trainingRuns).toHaveLength(1);
    expect(typeof result.current.onTrainClick).toBe("function");
    expect(typeof result.current.clearTrainingRuns).toBe("function");
  });
});
