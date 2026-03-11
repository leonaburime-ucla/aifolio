import { describe, expect, it, vi } from "vitest";
import { render, within } from "@testing-library/react";
import type { AssistantMessageProps } from "@copilotkit/react-ui";

vi.mock("@copilotkit/react-ui", () => ({
  AssistantMessage: ({ message }: { message: AssistantMessageProps["message"] }) => (
    <div data-testid="assistant-message">{typeof message?.content === "string" ? message.content : ""}</div>
  ),
}));

vi.mock("@/features/ag-ui-chat/typescript/logic/copilotAssistantPayload.util", () => ({
  extractCopilotDisplayMessage: (raw: string) => raw.replace(/^RAW:/, "DISPLAY:"),
  parseCopilotAssistantPayload: () => null,
}));

import CopilotAssistantMessage, {
  buildRenderedAssistantMessage,
  collectToolNamesFromUnknown,
  isStandaloneToolResultBlock,
  resolveToolNamesFromContent,
  resolveToolNamesFromMessage,
  stripStandaloneToolResultBlocks,
} from "@/features/ag-ui-chat/typescript/react/views/components/CopilotAssistantMessage";

describe("CopilotAssistantMessage helpers", () => {
  it("collects names from metadata and nested function names", () => {
    const names = new Set<string>();
    collectToolNamesFromUnknown(
      [
        { name: "set_pytorch_form_fields" },
        { function: { name: "start_pytorch_training_runs" } },
      ],
      names
    );

    expect(Array.from(names)).toEqual([
      "set_pytorch_form_fields",
      "start_pytorch_training_runs",
    ]);
  });

  it("resolves known tool names from free-form content", () => {
    const tools = resolveToolNamesFromContent(
      "Ran set_pytorch_form_fields then start_pytorch_training_runs."
    );
    expect(tools).toEqual([
      "set_pytorch_form_fields",
      "start_pytorch_training_runs",
    ]);
  });

  it("resolves tool names from message metadata collections", () => {
    const message = {
      id: "m1",
      role: "assistant",
      content: "RAW:hello",
      actions: [{ tool_name: "train_pytorch_model" }],
    } as AssistantMessageProps["message"];

    const tools = resolveToolNamesFromMessage(message);
    expect(tools).toEqual(["train_pytorch_model"]);
  });

  it("builds rendered message content with deduped tools", () => {
    const message = {
      id: "m2",
      role: "assistant",
      content:
        "RAW:done set_pytorch_form_fields and set_pytorch_form_fields",
      actions: [{ name: "set_pytorch_form_fields" }],
    } as AssistantMessageProps["message"];

    const rendered = buildRenderedAssistantMessage(message) as AssistantMessageProps["message"];
    expect(rendered?.content).toBe(
      "DISPLAY:done set_pytorch_form_fields and set_pytorch_form_fields\n\nTools used: set_pytorch_form_fields"
    );
  });

  it("identifies standalone tool-result payload blocks", () => {
    expect(
      isStandaloneToolResultBlock(
        '{"status":"ok","applied":["target_column"],"skipped":[],"via":"bridge"}'
      )
    ).toBe(true);
    expect(isStandaloneToolResultBlock("Assistant summary text")).toBe(false);
  });

  it("strips standalone tool-result payload blocks from assistant content", () => {
    const content = [
      "Changed the target column and randomized TensorFlow fields.",
      '{"status":"ok","applied":["target_column"],"skipped":[],"via":"bridge"}',
      '{"status":"ok","randomized":true,"patch":{"task":"auto"},"applied":["task"],"skipped":["training_mode"]}',
      '{"status":"ok","tab":"tensorflow"}',
    ].join("\n\n");

    expect(stripStandaloneToolResultBlocks(content)).toBe(
      "Changed the target column and randomized TensorFlow fields."
    );
  });
});

describe("CopilotAssistantMessage component", () => {
  it("renders transformed content with tool trace", () => {
    const view = render(
      <CopilotAssistantMessage
        message={{
          id: "m3",
          role: "assistant",
          content: "RAW:ok start_pytorch_training_runs",
        } as AssistantMessageProps["message"]}
      />
    );

    expect(within(view.container).getByTestId("assistant-message").textContent).toBe(
      "DISPLAY:ok start_pytorch_training_runs\n\nTools used: start_pytorch_training_runs"
    );
  });

  it("hides raw tool-result JSON blocks but keeps tool trace", () => {
    const view = render(
      <CopilotAssistantMessage
        message={{
          id: "m4",
          role: "assistant",
          actions: [{ name: "change_tensorflow_target_column" }],
          content: [
            "RAW:Changed the target column.",
            '{"status":"ok","applied":["target_column"],"skipped":[],"via":"bridge"}',
            '{"status":"ok","tab":"tensorflow"}',
          ].join("\n\n"),
        } as AssistantMessageProps["message"]}
      />
    );

    expect(within(view.container).getByTestId("assistant-message").textContent).toBe(
      "DISPLAY:Changed the target column.\n\nTools used: change_tensorflow_target_column"
    );
  });

  it("suppresses later fallback retries after a successful training-mode update", () => {
    const view = render(
      <CopilotAssistantMessage
        message={{
          id: "m5",
          role: "assistant",
          actions: [{ name: "set_active_ml_form_fields" }],
          content: [
            "RAW:Switched training mode to TabResNet.",
            "Updated PyTorch fields: training mode.",
            "Unable to update PyTorch form fields: NO_FIELDS_APPLIED.",
            "Let me try setting the model type to TabResNet.",
            "Unable to update PyTorch form fields: NO_FIELDS_APPLIED.",
          ].join("\n\n"),
        } as AssistantMessageProps["message"]}
      />
    );

    expect(within(view.container).getByTestId("assistant-message").textContent).toBe(
      "DISPLAY:Switched training mode to TabResNet.\n\nUpdated PyTorch fields: training mode.\n\nTools used: set_active_ml_form_fields"
    );
  });
});
