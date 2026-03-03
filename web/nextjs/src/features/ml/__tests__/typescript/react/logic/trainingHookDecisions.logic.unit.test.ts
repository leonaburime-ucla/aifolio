import { describe, expect, it } from "vitest";
import {
  calculatePlannedRunCount,
  hasTeacherModelReference,
  hasValidSweepInputs,
  isCompletedRunForMode,
} from "@/features/ml/typescript/react/logic/trainingHookDecisions.logic";

const valid = {
  epochsValidation: { ok: true as const, values: [60, 80] },
  testSizesValidation: { ok: true as const, values: [0.2] },
  learningRatesValidation: { ok: true as const, values: [0.001, 0.01] },
  batchSizesValidation: { ok: true as const, values: [64] },
  hiddenDimsValidation: { ok: true as const, values: [128, 256] },
  numHiddenLayersValidation: { ok: true as const, values: [2] },
  dropoutsValidation: { ok: true as const, values: [0.1, 0.2] },
};

describe("trainingHookDecisions.logic", () => {
  it("validates required sweep dimensions", () => {
    expect(hasValidSweepInputs({ isLinearBaselineMode: false, validations: valid })).toBe(true);
    expect(
      hasValidSweepInputs({
        isLinearBaselineMode: false,
        validations: { ...valid, epochsValidation: { ok: false as const, error: "bad" } },
      })
    ).toBe(false);
    expect(
      hasValidSweepInputs({
        isLinearBaselineMode: true,
        validations: {
          ...valid,
          hiddenDimsValidation: { ok: false as const, error: "bad" },
          numHiddenLayersValidation: { ok: false as const, error: "bad" },
          dropoutsValidation: { ok: false as const, error: "bad" },
        },
      })
    ).toBe(true);
    expect(
      hasValidSweepInputs({
        isLinearBaselineMode: false,
        validations: { ...valid, testSizesValidation: { ok: false as const, error: "bad" } },
      })
    ).toBe(false);
    expect(
      hasValidSweepInputs({
        isLinearBaselineMode: false,
        validations: { ...valid, learningRatesValidation: { ok: false as const, error: "bad" } },
      })
    ).toBe(false);
    expect(
      hasValidSweepInputs({
        isLinearBaselineMode: false,
        validations: { ...valid, batchSizesValidation: { ok: false as const, error: "bad" } },
      })
    ).toBe(false);
    expect(
      hasValidSweepInputs({
        isLinearBaselineMode: false,
        validations: { ...valid, hiddenDimsValidation: { ok: false as const, error: "bad" } },
      })
    ).toBe(false);
    expect(
      hasValidSweepInputs({
        isLinearBaselineMode: false,
        validations: { ...valid, numHiddenLayersValidation: { ok: false as const, error: "bad" } },
      })
    ).toBe(false);
    expect(
      hasValidSweepInputs({
        isLinearBaselineMode: false,
        validations: { ...valid, dropoutsValidation: { ok: false as const, error: "bad" } },
      })
    ).toBe(false);
  });

  it("calculates planned run count", () => {
    expect(calculatePlannedRunCount({ isLinearBaselineMode: false, validations: valid })).toBe(16);
    expect(
      calculatePlannedRunCount({
        isLinearBaselineMode: true,
        validations: valid,
      })
    ).toBe(4);
  });

  it("matches completed runs for active mode with numeric metrics", () => {
    expect(
      isCompletedRunForMode({
        run: { result: "completed", training_mode: "mlp_dense", metric_name: "accuracy", metric_score: "0.9" },
        mode: "mlp_dense",
      })
    ).toBe(true);
    expect(
      isCompletedRunForMode({
        run: { result: "failed", training_mode: "mlp_dense", metric_name: "accuracy", metric_score: "0.9" },
        mode: "mlp_dense",
      })
    ).toBe(false);
    expect(
      isCompletedRunForMode({
        run: { result: "completed", training_mode: "mlp_dense", metric_name: "n/a", metric_score: "n/a" },
        mode: "mlp_dense",
      })
    ).toBe(false);
    expect(
      isCompletedRunForMode({
        run: { result: "completed", training_mode: "wide_and_deep", metric_name: "accuracy", metric_score: "0.9" },
        mode: "mlp_dense",
      })
    ).toBe(false);
    expect(
      isCompletedRunForMode({
        run: { result: "completed", training_mode: "mlp_dense", metric_name: "accuracy", metric_score: "not-a-number" },
        mode: "mlp_dense",
      })
    ).toBe(false);
  });

  it("validates teacher model references", () => {
    expect(hasTeacherModelReference({ runId: "run-1", modelId: "", modelPath: "" })).toBe(true);
    expect(hasTeacherModelReference({ runId: "", modelId: "model-1", modelPath: "" })).toBe(true);
    expect(hasTeacherModelReference({ runId: "", modelId: "", modelPath: "/tmp/m" })).toBe(true);
    expect(hasTeacherModelReference({ runId: "n/a", modelId: "n/a", modelPath: "n/a" })).toBe(false);
  });
});
