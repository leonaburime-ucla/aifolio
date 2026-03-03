import { describe, expect, it } from "vitest";
import {
  buildSweepCombinations,
  validateBatchSizes,
  validateDropouts,
  validateEpochValues,
  validateHiddenDims,
  validateLearningRates,
  validateNumHiddenLayers,
  validateTestSizes,
} from "@/features/ml/typescript/validators/trainingSweep.validators";

describe("trainingSweep.validators", () => {
  it("validates epoch values", () => {
    expect(validateEpochValues({ raw: "10, 20" })).toEqual({ ok: true, values: [10, 20] });
    expect(validateEpochValues({ raw: "0" }).ok).toBe(false);
    expect(validateEpochValues({ raw: "1.5" }).ok).toBe(false);
  });

  it("validates test sizes and learning rates", () => {
    expect(validateTestSizes({ raw: "0.2 0.3" })).toEqual({ ok: true, values: [0.2, 0.3] });
    expect(validateTestSizes({ raw: "1" }).ok).toBe(false);
    expect(validateLearningRates({ raw: "0.001,0.01" })).toEqual({ ok: true, values: [0.001, 0.01] });
    expect(validateLearningRates({ raw: "0" }).ok).toBe(false);
  });

  it("validates integer sweep fields", () => {
    expect(validateBatchSizes({ raw: "32,64" })).toEqual({ ok: true, values: [32, 64] });
    expect(validateBatchSizes({ raw: "201" }).ok).toBe(false);
    expect(validateHiddenDims({ raw: "64,128" })).toEqual({ ok: true, values: [64, 128] });
    expect(validateHiddenDims({ raw: "7" }).ok).toBe(false);
    expect(validateNumHiddenLayers({ raw: "2,3" })).toEqual({ ok: true, values: [2, 3] });
    expect(validateNumHiddenLayers({ raw: "16" }).ok).toBe(false);
  });

  it("validates dropouts", () => {
    expect(validateDropouts({ raw: "0,0.2,0.9" })).toEqual({ ok: true, values: [0, 0.2, 0.9] });
    expect(validateDropouts({ raw: "-0.1" }).ok).toBe(false);
    expect(validateDropouts({ raw: "1" }).ok).toBe(false);
  });

  it("builds cartesian sweep combinations", () => {
    const out = buildSweepCombinations({
      config: {
        epochs: [10, 20],
        testSizes: [0.2],
        learningRates: [0.001],
        batchSizes: [32],
        hiddenDims: [64],
        numHiddenLayers: [2],
        dropouts: [0.1, 0.2],
      },
    });

    expect(out).toHaveLength(4);
    expect(out[0]).toEqual({
      epochs: 10,
      testSize: 0.2,
      learningRate: 0.001,
      batchSize: 32,
      hiddenDim: 64,
      numHiddenLayers: 2,
      dropout: 0.1,
    });
  });

  it("returns parse and integer validation errors", () => {
    expect(validateEpochValues({ raw: "" })).toEqual({
      ok: false,
      error: "Provide at least one value.",
    });
    expect(validateEpochValues({ raw: "abc" })).toEqual({
      ok: false,
      error: "Invalid number: abc",
    });
    expect(validateBatchSizes({ raw: "10.5" })).toEqual({
      ok: false,
      error: "Batch size must be an integer: 10.5",
    });
    expect(validateHiddenDims({ raw: "12.2" })).toEqual({
      ok: false,
      error: "Hidden dim must be an integer: 12.2",
    });
    expect(validateNumHiddenLayers({ raw: "2.5" })).toEqual({
      ok: false,
      error: "Hidden layers must be an integer: 2.5",
    });
  });
});
