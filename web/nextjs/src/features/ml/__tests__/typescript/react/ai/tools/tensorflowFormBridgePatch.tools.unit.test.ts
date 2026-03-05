import { describe, expect, it, vi } from "vitest";
import { applyTensorflowBridgePatch } from "@/features/ml/typescript/react/ai/tools/tensorflowFormBridgePatch.tools";
import type { TensorflowBridgePatchBindings } from "@/features/ml/__types__/typescript/react/ai/tools/tensorflowFormBridge.tools.types";

function createBindings(overrides: Partial<TensorflowBridgePatchBindings> = {}): TensorflowBridgePatchBindings {
  return {
    setTrainingMode: vi.fn(),
    setTargetColumn: vi.fn(),
    setTask: vi.fn(),
    runSweepEnabled: false,
    toggleRunSweep: vi.fn(),
    setEpochValuesInput: vi.fn(),
    setBatchSizesInput: vi.fn(),
    setLearningRatesInput: vi.fn(),
    setTestSizesInput: vi.fn(),
    setHiddenDimsInput: vi.fn(),
    setNumHiddenLayersInput: vi.fn(),
    setDropoutsInput: vi.fn(),
    setExcludeColumnsInput: vi.fn(),
    setDateColumnsInput: vi.fn(),
    autoDistillEnabled: false,
    setAutoDistillEnabled: vi.fn(),
    ...overrides,
  };
}

describe("tensorflowFormBridgePatch.tools", () => {
  it("applies patch deterministically and reports skipped keys", () => {
    const bindings = createBindings();
    const result = applyTensorflowBridgePatch(
      {
        training_mode: "wide_and_deep",
        epoch_values: [10, 20],
        run_sweep: true,
        auto_distill: true,
        unknown_key: "x",
      } as never,
      bindings
    );

    expect(bindings.setTrainingMode).toHaveBeenCalledWith("wide_and_deep");
    expect(bindings.setEpochValuesInput).toHaveBeenCalledWith("10,20");
    expect(bindings.toggleRunSweep).toHaveBeenCalledWith(true);
    expect(bindings.setAutoDistillEnabled).toHaveBeenCalledWith(true);
    expect(result.applied).toEqual(
      expect.arrayContaining(["training_mode", "epoch_values", "run_sweep", "auto_distill"])
    );
    expect(result.skipped).toContain("unknown_key");
  });

  it("does not toggle sweep/distill when values already match", () => {
    const bindings = createBindings({
      runSweepEnabled: true,
      autoDistillEnabled: true,
    });

    applyTensorflowBridgePatch(
      {
        run_sweep: true,
        auto_distill: true,
      },
      bindings
    );

    expect(bindings.toggleRunSweep).not.toHaveBeenCalled();
    expect(bindings.setAutoDistillEnabled).not.toHaveBeenCalled();
  });
});
