import { describe, expect, it, vi } from "vitest";
import {
  applyNumericInputs,
  buildRandomSweepInputs,
  metricHigherIsBetter,
  parseNumericValue,
} from "@/features/ml/typescript/utils/trainingUiShared";

describe("trainingUiShared", () => {
  it("builds randomized sweep inputs with deterministic random provider", () => {
    const random = vi.fn(() => 0.5);
    const snapshot = buildRandomSweepInputs({}, { random });

    expect(snapshot.epochValuesInput).toContain(",");
    expect(snapshot.batchSizesInput).toContain(",");
    expect(snapshot.learningRatesInput).toContain(",");
    expect(snapshot.testSizesInput).toContain(",");
    expect(snapshot.hiddenDimsInput).toContain(",");
    expect(snapshot.numHiddenLayersInput).toContain(",");
    expect(snapshot.dropoutsInput).toContain(",");
  });

  it("applies numeric snapshot to all setters", () => {
    const setters = {
      setEpochValuesInput: vi.fn(),
      setBatchSizesInput: vi.fn(),
      setLearningRatesInput: vi.fn(),
      setTestSizesInput: vi.fn(),
      setHiddenDimsInput: vi.fn(),
      setNumHiddenLayersInput: vi.fn(),
      setDropoutsInput: vi.fn(),
    };

    const snapshot = {
      epochValuesInput: "60,120",
      batchSizesInput: "32,64",
      learningRatesInput: "0.001,0.01",
      testSizesInput: "0.2,0.3",
      hiddenDimsInput: "64,128",
      numHiddenLayersInput: "2,3",
      dropoutsInput: "0.1,0.2",
    };

    applyNumericInputs({ snapshot, setters });
    expect(setters.setEpochValuesInput).toHaveBeenCalledWith("60,120");
    expect(setters.setDropoutsInput).toHaveBeenCalledWith("0.1,0.2");
  });

  it("parses numeric values including x10^ scientific notation", () => {
    expect(parseNumericValue({ value: 1.25 })).toBe(1.25);
    expect(parseNumericValue({ value: "1.2x10^3" })).toBe(1200);
    expect(parseNumericValue({ value: "   " })).toBeNull();
    expect(parseNumericValue({ value: "nope" })).toBeNull();
  });

  it("detects higher-is-better metrics", () => {
    expect(metricHigherIsBetter({ metricName: "accuracy" })).toBe(true);
    expect(metricHigherIsBetter({ metricName: "rmse" })).toBe(false);
  });
});
