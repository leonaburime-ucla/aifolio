import { describe, expect, it } from "vitest";
import { findOptimalParamsFromRuns } from "@/features/ml/typescript/utils/bayesianOptimizer.util";

function makeRun(overrides: Record<string, unknown> = {}) {
  return {
    result: "completed",
    metric_name: "accuracy",
    metric_score: "0.8",
    epochs: "60",
    learning_rate: "0.001",
    test_size: "0.2",
    batch_size: "64",
    hidden_dim: "128",
    num_hidden_layers: "2",
    dropout: "0.1",
    ...overrides,
  };
}

describe("bayesianOptimizer.util", () => {
  it("returns null when fewer than five valid completed runs", () => {
    const result = findOptimalParamsFromRuns({ rows: [makeRun(), makeRun()] });
    expect(result).toBeNull();
  });

  it("returns suggestion for valid completed runs with deterministic random", () => {
    const rows = [
      makeRun({ metric_score: 0.71, epochs: 40, learning_rate: 0.001 }),
      makeRun({ metric_score: 0.74, epochs: 50, learning_rate: 0.002 }),
      makeRun({ metric_score: 0.78, epochs: 55, learning_rate: 0.0008 }),
      makeRun({ metric_score: 0.81, epochs: 60, learning_rate: 0.0012 }),
      makeRun({ metric_score: 0.85, epochs: 70, learning_rate: 0.0015 }),
      makeRun({ metric_score: 0.83, epochs: 65, learning_rate: 0.0011 }),
      { ...makeRun({ metric_name: "n/a" }), result: "failed" },
    ];

    const randomValues = [0.17, 0.42, 0.88, 0.53, 0.29];
    let idx = 0;
    const result = findOptimalParamsFromRuns(
      { rows },
      { random: () => randomValues[idx++ % randomValues.length] }
    );

    expect(result).not.toBeNull();
    expect(result?.basedOnRuns).toBe(6);
    expect(result?.predictedMetricName).toBe("accuracy");
    expect(result?.suggestion.epochs).toBeGreaterThanOrEqual(1);
    expect(result?.suggestion.dropout).toBeGreaterThanOrEqual(0);
    expect(result?.suggestion.dropout).toBeLessThanOrEqual(0.9);
  });

  it("handles lower-is-better metrics and scientific notation", () => {
    const rows = [
      makeRun({ metric_name: "rmse", metric_score: "0.41", learning_rate: "1x10^-3" }),
      makeRun({ metric_name: "rmse", metric_score: "0.39", learning_rate: "2x10^-3" }),
      makeRun({ metric_name: "rmse", metric_score: "0.37", learning_rate: "3x10^-3" }),
      makeRun({ metric_name: "rmse", metric_score: "0.35", learning_rate: "4x10^-3" }),
      makeRun({ metric_name: "rmse", metric_score: "0.33", learning_rate: "5x10^-3" }),
      makeRun({ metric_name: "rmse", metric_score: "0.31", learning_rate: "6x10^-3" }),
    ];

    const result = findOptimalParamsFromRuns({ rows }, { random: () => 0.5 });
    expect(result).not.toBeNull();
    expect(result?.predictedMetricName).toBe("rmse");
  });
});
