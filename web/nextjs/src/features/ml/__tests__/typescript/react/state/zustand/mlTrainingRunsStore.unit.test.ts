import { beforeEach, describe, expect, it } from "vitest";
import { useMlTrainingRunsStore } from "@/features/ml/typescript/react/state/zustand/mlTrainingRunsStore";

describe("mlTrainingRunsStore", () => {
  beforeEach(() => {
    useMlTrainingRunsStore.setState({ trainingRuns: [] });
  });

  it("sets, prepends and clears runs", () => {
    useMlTrainingRunsStore.getState().setTrainingRuns([{ run_id: "r1" }]);
    expect(useMlTrainingRunsStore.getState().trainingRuns).toHaveLength(1);

    useMlTrainingRunsStore.getState().prependTrainingRun({ run_id: "r0" });
    expect(useMlTrainingRunsStore.getState().trainingRuns[0]?.run_id).toBe("r0");

    useMlTrainingRunsStore.getState().clearTrainingRuns();
    expect(useMlTrainingRunsStore.getState().trainingRuns).toEqual([]);
  });
});
