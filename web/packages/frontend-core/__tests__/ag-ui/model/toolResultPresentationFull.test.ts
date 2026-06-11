import { describe, expect, it } from "vitest";
import {
  humanizeFieldName,
  formatFieldList,
  getErrorCode,
  formatAddChartSpecToolResult,
  formatClearChartsToolResult,
  formatNavigateToPageToolResult,
  formatSwitchAgUiTabToolResult,
  formatSetFormFieldsToolResult,
  formatChangeTargetColumnToolResult,
  formatRandomizeFormFieldsToolResult,
  formatStartTrainingRunsToolResult,
  formatTrainModelToolResult,
} from "../../../src/ag-ui/model/toolResultPresentation";

describe("humanizeFieldName", () => {
  it("replaces underscores with spaces", () => {
    expect(humanizeFieldName("epoch_values")).toBe("epoch values");
    expect(humanizeFieldName("learning_rate")).toBe("learning rate");
  });

  it("handles fields without underscores", () => {
    expect(humanizeFieldName("epochs")).toBe("epochs");
  });
});

describe("formatFieldList", () => {
  it("joins humanized field names with commas", () => {
    expect(formatFieldList(["epoch_values", "batch_sizes"])).toBe("epoch values, batch sizes");
  });

  it("handles single field", () => {
    expect(formatFieldList(["target_column"])).toBe("target column");
  });
});

describe("getErrorCode", () => {
  it("returns code from object", () => {
    expect(getErrorCode({ code: "INVALID_CHART_SPEC" })).toBe("INVALID_CHART_SPEC");
  });

  it("returns UNKNOWN_ERROR for null/undefined", () => {
    expect(getErrorCode(null)).toBe("UNKNOWN_ERROR");
    expect(getErrorCode(undefined)).toBe("UNKNOWN_ERROR");
  });

  it("returns UNKNOWN_ERROR for missing or empty code", () => {
    expect(getErrorCode({})).toBe("UNKNOWN_ERROR");
    expect(getErrorCode({ code: "" })).toBe("UNKNOWN_ERROR");
    expect(getErrorCode({ code: "   " })).toBe("UNKNOWN_ERROR");
  });
});

describe("formatAddChartSpecToolResult", () => {
  it("singular for 1 chart", () => {
    expect(formatAddChartSpecToolResult({ status: "ok", addedCount: 1 })).toBe("Added 1 chart.");
  });

  it("plural for multiple charts", () => {
    expect(formatAddChartSpecToolResult({ status: "ok", addedCount: 3 })).toBe("Added 3 charts.");
  });

  it("error with code", () => {
    expect(formatAddChartSpecToolResult({ status: "error", code: "INVALID_CHART_SPEC" }))
      .toBe("Unable to add chart: INVALID_CHART_SPEC.");
  });
});

describe("formatClearChartsToolResult", () => {
  it("returns static message", () => {
    expect(formatClearChartsToolResult()).toBe("Cleared charts.");
  });
});

describe("formatNavigateToPageToolResult", () => {
  it("success with resolved route", () => {
    expect(formatNavigateToPageToolResult({ status: "ok", resolvedRoute: "/chat" }))
      .toBe("Navigated to /chat.");
  });

  it("error with code", () => {
    expect(formatNavigateToPageToolResult({ status: "error", code: "INVALID_ROUTE", allowedRoutes: [] }))
      .toBe("Unable to navigate: INVALID_ROUTE.");
  });
});

describe("formatSwitchAgUiTabToolResult", () => {
  it("success with tab name", () => {
    expect(formatSwitchAgUiTabToolResult({ status: "ok", tab: "charts" }))
      .toBe("Switched to the charts tab.");
  });

  it("error with code", () => {
    expect(formatSwitchAgUiTabToolResult({ status: "error", code: "INVALID_TAB", allowedTabs: [] }))
      .toBe("Unable to switch tabs: INVALID_TAB.");
  });
});

describe("formatSetFormFieldsToolResult", () => {
  it("success with applied fields", () => {
    expect(formatSetFormFieldsToolResult("PyTorch", { status: "ok", applied: ["epoch_values", "batch_sizes"] }))
      .toBe("Updated PyTorch fields: epoch values, batch sizes.");
  });

  it("success with no applied fields", () => {
    expect(formatSetFormFieldsToolResult("PyTorch", { status: "ok", applied: [] }))
      .toBe("Updated PyTorch form fields.");
  });

  it("success without applied key", () => {
    expect(formatSetFormFieldsToolResult("PyTorch", { status: "ok" }))
      .toBe("Updated PyTorch form fields.");
  });

  it("error", () => {
    expect(formatSetFormFieldsToolResult("TensorFlow", { status: "error", code: "FORM_NOT_READY" }))
      .toBe("Unable to update TensorFlow form fields: FORM_NOT_READY.");
  });
});

describe("formatChangeTargetColumnToolResult", () => {
  it("success with column name", () => {
    expect(formatChangeTargetColumnToolResult("PyTorch", "price", { status: "ok" }))
      .toBe("Changed PyTorch target column to price.");
  });

  it("success without column name", () => {
    expect(formatChangeTargetColumnToolResult("PyTorch", "", { status: "ok" }))
      .toBe("Changed PyTorch target column.");
  });

  it("success with undefined column", () => {
    expect(formatChangeTargetColumnToolResult("PyTorch", undefined, { status: "ok" }))
      .toBe("Changed PyTorch target column.");
  });

  it("error", () => {
    expect(formatChangeTargetColumnToolResult("PyTorch", "price", { status: "error", code: "NO_DATASET" }))
      .toBe("Unable to change PyTorch target column: NO_DATASET.");
  });
});

describe("formatRandomizeFormFieldsToolResult", () => {
  it("success", () => {
    expect(formatRandomizeFormFieldsToolResult("TensorFlow", { status: "ok" }))
      .toBe("Randomized TensorFlow form fields.");
  });

  it("error", () => {
    expect(formatRandomizeFormFieldsToolResult("TensorFlow", { status: "error", code: "X" }))
      .toBe("Unable to randomize TensorFlow form fields: X.");
  });
});

describe("formatStartTrainingRunsToolResult", () => {
  it("success", () => {
    expect(formatStartTrainingRunsToolResult("PyTorch", { status: "ok" }))
      .toBe("Started PyTorch training runs.");
  });

  it("error", () => {
    expect(formatStartTrainingRunsToolResult("PyTorch", { status: "error", code: "BUSY" }))
      .toBe("Unable to start PyTorch training runs: BUSY.");
  });
});

describe("formatTrainModelToolResult", () => {
  it("success with run_id", () => {
    expect(formatTrainModelToolResult("PyTorch", { status: "ok", run_id: "run-123" }))
      .toBe("Started one PyTorch training run (run-123).");
  });

  it("success without run_id", () => {
    expect(formatTrainModelToolResult("PyTorch", { status: "ok" }))
      .toBe("Started one PyTorch training run.");
  });

  it("success with empty run_id", () => {
    expect(formatTrainModelToolResult("PyTorch", { status: "ok", run_id: "   " }))
      .toBe("Started one PyTorch training run.");
  });

  it("error", () => {
    expect(formatTrainModelToolResult("TensorFlow", { status: "error", code: "TIMEOUT" }))
      .toBe("Unable to start TensorFlow training: TIMEOUT.");
  });
});
