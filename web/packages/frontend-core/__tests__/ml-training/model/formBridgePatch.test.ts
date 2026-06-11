import { describe, expect, it, vi } from "vitest";
import {
  applyPytorchBridgePatch,
  applyTensorflowBridgePatch,
  toBridgeCsv,
} from "../../../src/ml-training";
import type {
  PytorchBridgePatchBindings,
  TensorflowBridgePatchBindings,
} from "@aifolio/contracts/entities/ml-training";

function createPytorchBindings(
  overrides: Partial<PytorchBridgePatchBindings> = {}
): PytorchBridgePatchBindings {
  return {
    setDatasetId: vi.fn(),
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

function createTensorflowBindings(
  overrides: Partial<TensorflowBridgePatchBindings> = {}
): TensorflowBridgePatchBindings {
  return {
    setDatasetId: vi.fn(),
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
    autoDistillEnabled: true,
    setAutoDistillEnabled: vi.fn(),
    ...overrides,
  };
}

describe("ML form bridge patch core", () => {
  it("normalizes array and scalar values to form CSV strings", () => {
    expect(toBridgeCsv([" a ", "", 2])).toBe("a,2");
    expect(toBridgeCsv(null)).toBe("");
    expect(toBridgeCsv(" target ")).toBe("target");
  });

  it("applies PyTorch patches through injected bindings and reports unknown keys", () => {
    const bindings = createPytorchBindings();

    const result = applyPytorchBridgePatch(
      {
        dataset_id: "d1.csv",
        training_mode: "tabresnet",
        target_column: "churn",
        task: "classification",
        epoch_values: [20, 40],
        run_sweep: true,
        extra_field: "ignored",
      } as never,
      bindings
    );

    expect(bindings.setDatasetId).toHaveBeenCalledWith("d1.csv");
    expect(bindings.setTrainingMode).toHaveBeenCalledWith("tabresnet");
    expect(bindings.setTargetColumn).toHaveBeenCalledWith("churn");
    expect(bindings.setTask).toHaveBeenCalledWith("classification");
    expect(bindings.setEpochValuesInput).toHaveBeenCalledWith("20,40");
    expect(bindings.toggleRunSweep).toHaveBeenCalledWith(true);
    expect(result.applied).toEqual([
      "dataset_id",
      "training_mode",
      "target_column",
      "task",
      "run_sweep",
      "epoch_values",
    ]);
    expect(result.skipped).toEqual(["extra_field"]);
  });

  it("applies TensorFlow patches without toggling already-matching booleans", () => {
    const bindings = createTensorflowBindings({
      runSweepEnabled: true,
      autoDistillEnabled: true,
    });

    const result = applyTensorflowBridgePatch(
      {
        training_mode: "wide_and_deep",
        run_sweep: true,
        auto_distill: true,
      },
      bindings
    );

    expect(bindings.setTrainingMode).toHaveBeenCalledWith("wide_and_deep");
    expect(bindings.toggleRunSweep).not.toHaveBeenCalled();
    expect(bindings.setAutoDistillEnabled).not.toHaveBeenCalled();
    expect(result.skipped).toEqual([]);
  });
});
