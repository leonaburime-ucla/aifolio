import { renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { usePytorchTrainingIntegration } from "@/features/ml/typescript/react/hooks/usePytorchTraining.hooks";

describe("usePytorchTrainingIntegration", () => {
  it("combines dataset/ui/logic/training runs", () => {
    const { result } = renderHook(() =>
      usePytorchTrainingIntegration({
        useDatasetState: () => ({
          datasetOptions: [],
          selectedDatasetId: "d1",
          setSelectedDatasetId: vi.fn(),
          isLoading: false,
          error: null,
          tableRows: [],
          tableColumns: ["target"],
          rowCount: 0,
          totalRowCount: 0,
          reloadManifest: vi.fn(async () => undefined),
          reloadDataset: vi.fn(async () => undefined),
        }),
        useTrainingRunsState: () => ({
          trainingRuns: [{ run_id: "r1" }],
          prependTrainingRun: vi.fn(),
          clearTrainingRuns: vi.fn(),
        }),
        trainModel: vi.fn(async () => ({ status: "ok" })),
        distillModel: vi.fn(async () => ({ status: "ok" })),
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
