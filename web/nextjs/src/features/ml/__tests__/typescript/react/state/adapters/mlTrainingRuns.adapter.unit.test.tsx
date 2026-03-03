import { beforeEach, describe, expect, it } from "vitest";
import { renderHook } from "@testing-library/react";
import { useMlTrainingRunsAdapter } from "@/features/ml/typescript/react/state/adapters/mlTrainingRuns.adapter";
import { useMlTrainingRunsStore } from "@/features/ml/typescript/react/state/zustand/mlTrainingRunsStore";

describe("mlTrainingRuns.adapter", () => {
  beforeEach(() => {
    useMlTrainingRunsStore.setState({ trainingRuns: [] });
  });

  it("returns training runs and actions", () => {
    const { result } = renderHook(() => useMlTrainingRunsAdapter());
    expect(result.current.trainingRuns).toEqual([]);

    result.current.prependTrainingRun({ run_id: "r1" });
    expect(useMlTrainingRunsStore.getState().trainingRuns[0]?.run_id).toBe("r1");

    result.current.clearTrainingRuns();
    expect(useMlTrainingRunsStore.getState().trainingRuns).toEqual([]);
  });
});
