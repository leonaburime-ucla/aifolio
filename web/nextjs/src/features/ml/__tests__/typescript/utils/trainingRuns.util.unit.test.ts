import { describe, expect, it } from "vitest";
import {
  calcTrainingTableHeight,
  formatCompletedAt,
  formatMetricNumber,
} from "@/features/ml/typescript/utils/trainingRuns.util";

describe("trainingRuns.util", () => {
  it("formats completed timestamp deterministically for provided date", () => {
    const date = new Date("2026-03-03T12:34:56");
    expect(formatCompletedAt({ date })).toMatch(/^\d{2}\/\d{2}\/\d{2} \d{2}:\d{2}:\d{2}$/);
  });

  it("formats metric numbers across value ranges", () => {
    expect(formatMetricNumber({ value: 0 })).toBe("0");
    expect(formatMetricNumber({ value: Number.NaN })).toBe("n/a");
    expect(formatMetricNumber({ value: 0.0000009 })).toContain("x10^");
    expect(formatMetricNumber({ value: 5_000_000 })).toContain("x10^");
  });

  it("clamps training table height to min and max bounds", () => {
    expect(calcTrainingTableHeight({ rowsCount: 0 })).toBe(140);
    expect(calcTrainingTableHeight({ rowsCount: 3 })).toBe(208);
    expect(calcTrainingTableHeight({ rowsCount: 100 })).toBe(360);
  });
});
