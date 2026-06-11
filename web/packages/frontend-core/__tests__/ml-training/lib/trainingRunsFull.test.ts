import { describe, expect, it } from "vitest";
import {
  calcTrainingTableHeight,
  formatCompletedAt,
  formatMetricNumber,
  TRAINING_RUN_COLUMNS,
} from "../../../src/ml-training/lib/trainingRuns";

describe("TRAINING_RUN_COLUMNS", () => {
  it("exports a frozen array of column names", () => {
    expect(Array.isArray(TRAINING_RUN_COLUMNS)).toBe(true);
    expect(TRAINING_RUN_COLUMNS.length).toBeGreaterThan(10);
    expect(TRAINING_RUN_COLUMNS).toContain("completed_at");
    expect(TRAINING_RUN_COLUMNS).toContain("metric_score");
    expect(TRAINING_RUN_COLUMNS).toContain("training_mode");
  });
});

describe("formatCompletedAt", () => {
  it("formats date as MM/DD/YY HH:mm:ss", () => {
    const date = new Date(2026, 2, 3, 14, 5, 9);
    expect(formatCompletedAt({ date })).toBe("03/03/26 14:05:09");
  });

  it("pads single-digit month and day", () => {
    const date = new Date(2026, 0, 1, 1, 2, 3);
    expect(formatCompletedAt({ date })).toBe("01/01/26 01:02:03");
  });

  it("uses current time when date is omitted", () => {
    const result = formatCompletedAt({});
    expect(result).toMatch(/^\d{2}\/\d{2}\/\d{2} \d{2}:\d{2}:\d{2}$/);
  });
});

describe("formatMetricNumber", () => {
  it("returns n/a for non-number values", () => {
    expect(formatMetricNumber({ value: null })).toBe("n/a");
    expect(formatMetricNumber({ value: undefined })).toBe("n/a");
    expect(formatMetricNumber({ value: "0.5" })).toBe("n/a");
  });

  it("returns n/a for NaN", () => {
    expect(formatMetricNumber({ value: NaN })).toBe("n/a");
  });

  it("returns '0' for zero", () => {
    expect(formatMetricNumber({ value: 0 })).toBe("0");
  });

  it("uses scientific notation for very small values (abs < 1e-5)", () => {
    const result = formatMetricNumber({ value: 0.000003 });
    expect(result).toContain("x10^");
    expect(result).toContain("-6");
  });

  it("uses scientific notation for very large values (abs >= 1e6)", () => {
    const result = formatMetricNumber({ value: 2500000 });
    expect(result).toContain("x10^");
  });

  it("uses toPrecision(5) for normal range values", () => {
    expect(formatMetricNumber({ value: 0.91234 })).toBe("0.91234");
    expect(formatMetricNumber({ value: 123.4 })).toBe("123.4");
  });

  it("handles negative small values", () => {
    const result = formatMetricNumber({ value: -0.000001 });
    expect(result).toContain("x10^");
    expect(result).toContain("-");
  });

  it("handles negative large values", () => {
    const result = formatMetricNumber({ value: -5000000 });
    expect(result).toContain("x10^");
  });
});

describe("calcTrainingTableHeight", () => {
  it("returns minimum height for 0 rows", () => {
    expect(calcTrainingTableHeight({ rowsCount: 0 })).toBe(140);
  });

  it("returns minimum height for 1 row (64+48=112 < 140)", () => {
    expect(calcTrainingTableHeight({ rowsCount: 1 })).toBe(140);
  });

  it("returns computed height for moderate rows", () => {
    // 64 + 3*48 = 208
    expect(calcTrainingTableHeight({ rowsCount: 3 })).toBe(208);
  });

  it("clamps to maximum height for many rows", () => {
    expect(calcTrainingTableHeight({ rowsCount: 50 })).toBe(360);
  });
});
