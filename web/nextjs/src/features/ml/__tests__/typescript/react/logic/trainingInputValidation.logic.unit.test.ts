import { describe, expect, it } from "vitest";
import {
  resolveTargetColumn,
  resolveTeacherRunKey,
  splitColumnInput,
  validateTrainingSetup,
} from "@/features/ml/typescript/logic/trainingInputValidation.logic";

describe("trainingInputValidation.logic", () => {
  it("splits and normalizes comma-delimited column input", () => {
    expect(splitColumnInput({ value: "  a, b ,,c  " })).toEqual(["a", "b", "c"]);
  });

  it("resolves target column with explicit value first", () => {
    expect(
      resolveTargetColumn({
        targetColumn: " target ",
        defaultTargetColumn: "fallback",
        tableColumns: ["col1"],
      })
    ).toBe("target");
  });

  it("falls back to default or first table column for target resolution", () => {
    expect(
      resolveTargetColumn({
        targetColumn: " ",
        defaultTargetColumn: "fallback",
        tableColumns: ["col1"],
      })
    ).toBe("fallback");

    expect(
      resolveTargetColumn({
        targetColumn: "",
        defaultTargetColumn: "",
        tableColumns: ["col1"],
      })
    ).toBe("col1");

    expect(
      resolveTargetColumn({
        targetColumn: "",
        defaultTargetColumn: "",
        tableColumns: [],
      })
    ).toBe("");
  });

  it("validates training setup and returns first error", () => {
    const baseValidations = {
      epochsValidation: { ok: true as const, values: [60] },
      testSizesValidation: { ok: true as const, values: [0.2] },
      learningRatesValidation: { ok: true as const, values: [0.001] },
      batchSizesValidation: { ok: true as const, values: [64] },
      hiddenDimsValidation: { ok: true as const, values: [128] },
      numHiddenLayersValidation: { ok: true as const, values: [2] },
      dropoutsValidation: { ok: true as const, values: [0.1] },
    };

    expect(
      validateTrainingSetup({
        selectedDatasetId: null,
        resolvedTargetColumn: "churn",
        excludeColumns: [],
        dateColumns: [],
        isLinearBaselineMode: false,
        validations: baseValidations,
      })
    ).toBe("Please select a dataset first.");

    expect(
      validateTrainingSetup({
        selectedDatasetId: "dataset.csv",
        resolvedTargetColumn: "churn",
        excludeColumns: ["churn"],
        dateColumns: [],
        isLinearBaselineMode: false,
        validations: baseValidations,
      })
    ).toBe("Target column cannot also be in excluded columns.");

    expect(
      validateTrainingSetup({
        selectedDatasetId: "dataset.csv",
        resolvedTargetColumn: "churn",
        excludeColumns: [],
        dateColumns: ["churn"],
        isLinearBaselineMode: false,
        validations: baseValidations,
      })
    ).toBe("Target column cannot also be in date columns.");

    expect(
      validateTrainingSetup({
        selectedDatasetId: "dataset.csv",
        resolvedTargetColumn: "churn",
        excludeColumns: ["date_col"],
        dateColumns: ["date_col"],
        isLinearBaselineMode: false,
        validations: baseValidations,
      })
    ).toBe("Column 'date_col' cannot be in both excluded and date columns.");

    expect(
      validateTrainingSetup({
        selectedDatasetId: "dataset.csv",
        resolvedTargetColumn: " ",
        excludeColumns: [],
        dateColumns: [],
        isLinearBaselineMode: false,
        validations: baseValidations,
      })
    ).toBe("Please provide a target column.");
  });

  it("returns validator-specific errors in expected order", () => {
    const failEpochs = {
      epochsValidation: { ok: false as const, error: "epochs bad" },
      testSizesValidation: { ok: true as const, values: [0.2] },
      learningRatesValidation: { ok: true as const, values: [0.001] },
      batchSizesValidation: { ok: true as const, values: [64] },
      hiddenDimsValidation: { ok: true as const, values: [128] },
      numHiddenLayersValidation: { ok: true as const, values: [2] },
      dropoutsValidation: { ok: true as const, values: [0.1] },
    };
    expect(
      validateTrainingSetup({
        selectedDatasetId: "dataset.csv",
        resolvedTargetColumn: "target",
        excludeColumns: [],
        dateColumns: [],
        isLinearBaselineMode: false,
        validations: failEpochs,
      })
    ).toBe("epochs bad");

    const failDropout = {
      ...failEpochs,
      epochsValidation: { ok: true as const, values: [60] },
      dropoutsValidation: { ok: false as const, error: "dropout bad" },
    };
    expect(
      validateTrainingSetup({
        selectedDatasetId: "dataset.csv",
        resolvedTargetColumn: "target",
        excludeColumns: [],
        dateColumns: [],
        isLinearBaselineMode: false,
        validations: failDropout,
      })
    ).toBe("dropout bad");

    const failTestSizes = {
      ...failEpochs,
      epochsValidation: { ok: true as const, values: [60] },
      testSizesValidation: { ok: false as const, error: "test size bad" },
    };
    expect(
      validateTrainingSetup({
        selectedDatasetId: "dataset.csv",
        resolvedTargetColumn: "target",
        excludeColumns: [],
        dateColumns: [],
        isLinearBaselineMode: false,
        validations: failTestSizes,
      })
    ).toBe("test size bad");

    const failLearningRate = {
      ...failEpochs,
      epochsValidation: { ok: true as const, values: [60] },
      learningRatesValidation: { ok: false as const, error: "learning bad" },
    };
    expect(
      validateTrainingSetup({
        selectedDatasetId: "dataset.csv",
        resolvedTargetColumn: "target",
        excludeColumns: [],
        dateColumns: [],
        isLinearBaselineMode: false,
        validations: failLearningRate,
      })
    ).toBe("learning bad");

    const failBatch = {
      ...failEpochs,
      epochsValidation: { ok: true as const, values: [60] },
      batchSizesValidation: { ok: false as const, error: "batch bad" },
    };
    expect(
      validateTrainingSetup({
        selectedDatasetId: "dataset.csv",
        resolvedTargetColumn: "target",
        excludeColumns: [],
        dateColumns: [],
        isLinearBaselineMode: false,
        validations: failBatch,
      })
    ).toBe("batch bad");

    const failHidden = {
      ...failEpochs,
      epochsValidation: { ok: true as const, values: [60] },
      hiddenDimsValidation: { ok: false as const, error: "hidden bad" },
    };
    expect(
      validateTrainingSetup({
        selectedDatasetId: "dataset.csv",
        resolvedTargetColumn: "target",
        excludeColumns: [],
        dateColumns: [],
        isLinearBaselineMode: false,
        validations: failHidden,
      })
    ).toBe("hidden bad");

    const failLayers = {
      ...failEpochs,
      epochsValidation: { ok: true as const, values: [60] },
      numHiddenLayersValidation: { ok: false as const, error: "layers bad" },
    };
    expect(
      validateTrainingSetup({
        selectedDatasetId: "dataset.csv",
        resolvedTargetColumn: "target",
        excludeColumns: [],
        dateColumns: [],
        isLinearBaselineMode: false,
        validations: failLayers,
      })
    ).toBe("layers bad");
  });

  it("skips hidden/droupout-style checks in linear baseline mode", () => {
    const invalidDeepValidations = {
      epochsValidation: { ok: true as const, values: [60] },
      testSizesValidation: { ok: true as const, values: [0.2] },
      learningRatesValidation: { ok: true as const, values: [0.001] },
      batchSizesValidation: { ok: true as const, values: [64] },
      hiddenDimsValidation: { ok: false as const, error: "bad hidden" },
      numHiddenLayersValidation: { ok: false as const, error: "bad layers" },
      dropoutsValidation: { ok: false as const, error: "bad dropout" },
    };

    expect(
      validateTrainingSetup({
        selectedDatasetId: "dataset.csv",
        resolvedTargetColumn: "churn",
        excludeColumns: [],
        dateColumns: [],
        isLinearBaselineMode: true,
        validations: invalidDeepValidations,
      })
    ).toBeNull();
  });

  it("resolves a teacher key from run/model/path fields", () => {
    expect(resolveTeacherRunKey({ run: { run_id: "run-1" } })).toBe("run-1");
    expect(resolveTeacherRunKey({ run: { model_id: "model-1" } })).toBe("model-1");
    expect(resolveTeacherRunKey({ run: { model_path: "/tmp/model.pt" } })).toBe("/tmp/model.pt");
    expect(resolveTeacherRunKey({ run: { completed_at: "yesterday" } })).toBe("yesterday");
    expect(resolveTeacherRunKey({ run: {} })).toBe("run");
  });
});
