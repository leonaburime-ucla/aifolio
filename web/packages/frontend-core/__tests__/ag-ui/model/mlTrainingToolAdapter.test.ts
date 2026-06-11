import { describe, expect, it } from "vitest";
import {
  buildRandomPytorchFormPatch,
  buildRandomTensorflowFormPatch,
  resolveTargetColumnChangeFromOptions,
  selectSweepValues,
} from "../../../src/ag-ui";

describe("AG-UI ML training tool adapter core", () => {
  it("handles empty, invalid, single, and full sweep selection requests", () => {
    expect(selectSweepValues([], 2)).toEqual([]);
    expect(selectSweepValues([10, 20], Number.NaN, { random: () => 0 })).toEqual([10]);
    expect(selectSweepValues([10, 20], 1.9, { random: () => 0.9 })).toEqual([20]);
    expect(selectSweepValues([10, 20], 5, { random: () => 0 })).toEqual([10, 20]);
  });

  it("selects a bounded deterministic subset for sweep values", () => {
    const randomValues = [0.9, 0.1];
    const selected = selectSweepValues([10, 20, 30], 2, {
      random: () => randomValues.shift() ?? 0,
    });

    expect(selected).toEqual([30, 10]);
  });

  it("resolves explicit, next, and different target-column changes", () => {
    const options = [
      { value: "id" },
      { value: "income", isCurrent: true },
      { value: "churn" },
    ];

    expect(
      resolveTargetColumnChangeFromOptions(options, { target_column: "churn" })
    ).toBe("churn");
    expect(resolveTargetColumnChangeFromOptions(options, { mode: "next" })).toBe("churn");
    expect(
      resolveTargetColumnChangeFromOptions(options, { mode: "different" }, { random: () => 0 })
    ).toBe("id");
    expect(
      resolveTargetColumnChangeFromOptions(options, { target_column: "missing" })
    ).toBeNull();
    expect(resolveTargetColumnChangeFromOptions([], { mode: "next" })).toBeNull();
    expect(resolveTargetColumnChangeFromOptions([{ value: "   " }], { mode: "next" })).toBeNull();
    expect(resolveTargetColumnChangeFromOptions([{ value: "income" }], { mode: "next" })).toBe("income");
    expect(
      resolveTargetColumnChangeFromOptions([{ value: "income", isCurrent: true }], { mode: "different" })
    ).toBe("income");
  });

  it("builds PyTorch randomization patches from current form values", () => {
    const patch = buildRandomPytorchFormPatch(
      { value_count: 1 },
      {
        random: () => 0,
        getSelectValue: (field) =>
          field === "pytorch_training_mode"
            ? "linear_glm_baseline"
            : field === "pytorch_task"
              ? "classification"
              : null,
      }
    );

    expect(patch.training_mode).toBeUndefined();
    expect(patch.task).toBe("classification");
    expect(patch.hidden_dims).toBe("128");
    expect(patch.num_hidden_layers).toBe("2");
    expect(patch.dropouts).toBe("0.1");
    expect(patch.epoch_values).toEqual([30]);
  });

  it("builds TensorFlow randomization patches with optional model randomization", () => {
    const patch = buildRandomTensorflowFormPatch(
      { randomize_model_fields: true, value_count: 1 },
      {
        random: () => 0,
        getSelectValue: () => "regression",
      }
    );

    expect(patch.training_mode).toBe("mlp_dense");
    expect(patch.task).toBe("auto");
    expect(patch.batch_sizes).toEqual([32]);
  });

  it("builds style-specific PyTorch patches and randomizes model fields on request", () => {
    const aggressive = buildRandomPytorchFormPatch(
      { style: "aggressive", randomize_model_fields: true, value_count: 3, run_sweep: true },
      { random: () => 0.99 }
    );
    const safe = buildRandomPytorchFormPatch(
      { style: "safe", value_count: 2, set_sweep_values: false, auto_distill: true },
      {
        random: () => 0,
        getSelectValue: (field) =>
          field === "pytorch_training_mode"
            ? null
            : field === "pytorch_task"
              ? null
              : null,
      }
    );

    expect(aggressive.training_mode).toBe("tree_teacher_distillation");
    expect(aggressive.task).toBe("regression");
    expect(aggressive.epoch_values).toEqual([20, 40, 60]);
    expect(aggressive.run_sweep).toBe(true);
    expect(safe.training_mode).toBeUndefined();
    expect(safe.task).toBe("auto");
    expect(safe.epoch_values).toEqual([40, 80]);
    expect(safe.run_sweep).toBe(false);
    expect(safe.auto_distill).toBe(true);
  });

  it("builds style-specific TensorFlow patches and preserves task when model fields stay locked", () => {
    const safe = buildRandomTensorflowFormPatch(
      { style: "safe", value_count: 2, run_sweep: true, auto_distill: false },
      {
        random: () => 0,
        getSelectValue: (field) => (field === "tensorflow_task" ? "classification" : null),
      }
    );
    const aggressive = buildRandomTensorflowFormPatch(
      { style: "aggressive", randomize_model_fields: true, value_count: 3 },
      { random: () => 0.99 }
    );

    expect(safe.training_mode).toBeUndefined();
    expect(safe.task).toBe("classification");
    expect(safe.epoch_values).toEqual([40, 80]);
    expect(safe.run_sweep).toBe(true);
    expect(safe.auto_distill).toBe(false);
    expect(aggressive.training_mode).toBe("time_aware_tabular");
    expect(aggressive.task).toBe("regression");
    expect(aggressive.hidden_dims).toEqual([64, 128, 256]);
  });
});
