import { describe, expect, it } from "vitest";

import {
  formatChangeTargetColumnToolResult,
  formatNavigateToPageToolResult,
  formatSetFormFieldsToolResult,
  formatSwitchAgUiTabToolResult,
} from "@/features/ag-ui-chat/typescript/logic/copilotToolResultPresentation.logic";

describe("copilotToolResultPresentation.logic", () => {
  it("formats tab-switch results without leaking raw status payloads", () => {
    expect(formatSwitchAgUiTabToolResult({ status: "ok", tab: "tensorflow" })).toBe(
      "Switched to the tensorflow tab."
    );
  });

  it("formats navigation results without leaking raw status payloads", () => {
    expect(
      formatNavigateToPageToolResult({ status: "ok", resolvedRoute: "/agentic-research" })
    ).toBe("Navigated to /agentic-research.");
  });

  it("formats set-form results with humanized field names", () => {
    expect(
      formatSetFormFieldsToolResult("PyTorch", {
        status: "ok",
        applied: ["batch_sizes", "hidden_dims"],
      })
    ).toBe("Updated PyTorch fields: batch sizes, hidden dims.");
  });

  it("formats target-column changes with explicit target names", () => {
    expect(
      formatChangeTargetColumnToolResult(
        "TensorFlow",
        "revenue",
        { status: "ok", applied: ["target_column"] }
      )
    ).toBe("Changed TensorFlow target column to revenue.");
  });
});
