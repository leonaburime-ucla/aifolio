import { act, renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { useMlTrainingUiBaseState } from "@/features/ml/typescript/react/hooks/ml.hooks.base";

describe("ml.hooks.base", () => {
  it("provides default base state and setters", () => {
    const { result } = renderHook(() => useMlTrainingUiBaseState());

    expect(result.current.task).toBe("auto");
    expect(result.current.epochValuesInput).toBe("60");
    expect(result.current.runSweepEnabled).toBe(false);
    expect(result.current.trainingProgress).toEqual({ current: 0, total: 0 });

    act(() => {
      result.current.setTask("classification");
      result.current.setEpochValuesInput("99");
      result.current.setRunSweepEnabled(true);
      result.current.setTrainingError("boom");
    });

    expect(result.current.task).toBe("classification");
    expect(result.current.epochValuesInput).toBe("99");
    expect(result.current.runSweepEnabled).toBe(true);
    expect(result.current.trainingError).toBe("boom");
  });
});
