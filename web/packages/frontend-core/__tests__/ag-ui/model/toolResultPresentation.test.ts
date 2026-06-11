import { describe, expect, it } from "vitest";

import {
  formatChangeTargetColumnToolResult,
  formatNavigateToPageToolResult,
  formatSetFormFieldsToolResult,
  formatSwitchAgUiTabToolResult,
} from "@aifolio/frontend-core/ag-ui";

describe("copilotToolResultPresentation.logic", () => {
  it("formats tab-switch results without leaking raw status payloads", () => {
    expect(formatSwitchAgUiTabToolResult({ status: "ok" as const, tab: "tensorflow" })).toBe(
      "Switched to the tensorflow tab."
    );
  });

  it("formats navigation results without leaking raw status payloads", () => {
    expect(
      formatNavigateToPageToolResult({ status: "ok" as const, resolvedRoute: "/agentic-research" })
    ).toBe("Navigated to /agentic-research.");
  });

  it("formats set-form results with humanized field names", () => {
    expect(
      formatSetFormFieldsToolResult("PyTorch", {
        status: "ok" as const,
        applied: ["batch_sizes", "hidden_dims"],
      })
    ).toBe("Updated PyTorch fields: batch sizes, hidden dims.");
  });

  it("formats target-column changes with explicit target names", () => {
    expect(
      formatChangeTargetColumnToolResult(
        "TensorFlow",
        "revenue",
        { status: "ok" as const, applied: ["target_column"] }
      )
    ).toBe("Changed TensorFlow target column to revenue.");
  });
});
