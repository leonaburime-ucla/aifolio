"use client";
import type { AssistantMessageProps } from "@copilotkit/react-ui";
import { AssistantMessage as DefaultAssistantMessage } from "@copilotkit/react-ui";
import {
  extractCopilotDisplayMessage,
  parseCopilotAssistantPayload,
} from "@/features/ag-ui-chat/typescript/logic/copilotAssistantPayload.util";

const DEBUG_COPILOT =
  process.env.NEXT_PUBLIC_COPILOT_DEBUG === "1" ||
  process.env.COPILOT_DEBUG === "1";

/**
 * Canonical AG-UI tool names used for tool-trace extraction from assistant metadata/content.
 */
export const KNOWN_TOOL_NAMES = [
  "switch_ag_ui_tab",
  "set_active_ml_form_fields",
  "change_active_ml_target_column",
  "randomize_active_ml_form_fields",
  "start_active_ml_training_runs",
  "set_pytorch_form_fields",
  "change_pytorch_target_column",
  "randomize_pytorch_form_fields",
  "start_pytorch_training_runs",
  "train_pytorch_model",
  "set_tensorflow_form_fields",
  "change_tensorflow_target_column",
  "randomize_tensorflow_form_fields",
  "start_tensorflow_training_runs",
  "train_tensorflow_model",
  "add_chart_spec",
  "clear_charts",
  "navigate_to_page",
] as const;

/**
 * Collect tool names from unknown structured values into a mutable set.
 *
 * Supports:
 * - plain strings containing known tool names
 * - arrays of nested values
 * - objects carrying `name`, `toolName`, `tool_name`, or `function.name`
 *
 * @param value Candidate value that may contain tool-call metadata.
 * @param output Destination set for discovered tool names.
 */
export function collectToolNamesFromUnknown(value: unknown, output: Set<string>): void {
  if (!value) return;
  if (Array.isArray(value)) {
    value.forEach((entry) => collectToolNamesFromUnknown(entry, output));
    return;
  }
  if (typeof value === "string") {
    for (const name of KNOWN_TOOL_NAMES) {
      if (value.includes(name)) output.add(name);
    }
    return;
  }
  if (typeof value !== "object") return;

  const record = value as Record<string, unknown>;
  const directName =
    (typeof record.name === "string" && record.name) ||
    (typeof record.toolName === "string" && record.toolName) ||
    (typeof record.tool_name === "string" && record.tool_name);
  if (directName) output.add(directName);

  const fn = record.function;
  if (fn && typeof fn === "object") {
    const fnName = (fn as Record<string, unknown>).name;
    if (typeof fnName === "string" && fnName) output.add(fnName);
  }
}

/**
 * Resolve unique tool names from an assistant message object metadata.
 *
 * @param message Copilot assistant message object.
 * @returns Unique tool names discovered from message metadata fields.
 */
export function resolveToolNamesFromMessage(message: AssistantMessageProps["message"]): string[] {
  if (!message || typeof message !== "object") return [];
  const record = message as Record<string, unknown>;
  const names = new Set<string>();
  collectToolNamesFromUnknown(record.actions, names);
  collectToolNamesFromUnknown(record.toolCalls, names);
  collectToolNamesFromUnknown(record.tool_calls, names);
  collectToolNamesFromUnknown(record.tools, names);
  return Array.from(names);
}

/**
 * Resolve unique tool names referenced directly in assistant text content.
 *
 * @param rawContent Raw assistant content string.
 * @returns Unique tool names found in content.
 */
export function resolveToolNamesFromContent(rawContent: string): string[] {
  const names = new Set<string>();
  collectToolNamesFromUnknown(rawContent, names);
  return Array.from(names);
}

/**
 * Detect whether a paragraph is a raw tool-result JSON object that should stay
 * out of the visible assistant transcript.
 *
 * We keep the assistant's natural-language summary and `Tools used:` trace, but
 * hide low-signal structured payloads such as `{"status":"ok","applied":[...]}`.
 *
 * @param block Candidate paragraph/block.
 * @returns `true` when the block is a standalone tool-result payload.
 */
export function isStandaloneToolResultBlock(block: string): boolean {
  const trimmed = block.trim();
  if (!trimmed.startsWith("{") || !trimmed.endsWith("}")) return false;

  try {
    const parsed = JSON.parse(trimmed);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return false;
    const record = parsed as Record<string, unknown>;
    if (typeof record.status !== "string") return false;
    return (
      "applied" in record ||
      "skipped" in record ||
      "via" in record ||
      "randomized" in record ||
      "tab" in record ||
      "started" in record ||
      "code" in record
    );
  } catch {
    return false;
  }
}

/**
 * Remove standalone raw tool-result JSON blocks from assistant text while
 * preserving user-facing narrative text around them.
 *
 * @param content Assistant display content.
 * @returns Content with tool-result payload paragraphs removed.
 */
export function stripStandaloneToolResultBlocks(content: string): string {
  return content
    .split(/\n\s*\n+/)
    .map((block) => block.trim())
    .filter((block) => block.length > 0 && !isStandaloneToolResultBlock(block))
    .join("\n\n");
}

function isSuccessBlock(block: string): boolean {
  return (
    /^Switched training mode to /i.test(block) ||
    /^Updated (PyTorch|TensorFlow) fields:/i.test(block)
  );
}

function isPostSuccessRetryNoiseBlock(block: string): boolean {
  return (
    /NO_FIELDS_APPLIED/i.test(block) ||
    /^Let me try /i.test(block) ||
    /^It looks like /i.test(block) ||
    /^I'm having trouble finding the exact field/i.test(block)
  );
}

/**
 * Suppresses planner retry chatter that appears after a successful ML field
 * update. We keep the first successful narrative/result pair and drop later
 * "let me try again" blocks caused by stale retry planning.
 */
export function stripPostSuccessRetryNoise(content: string): string {
  const blocks = content
    .split(/\n\s*\n+/)
    .map((block) => block.trim())
    .filter((block) => block.length > 0);

  let hasSuccessfulMlUpdate = false;

  return blocks
    .filter((block) => {
      if (isSuccessBlock(block)) {
        hasSuccessfulMlUpdate = true;
        return true;
      }
      if (hasSuccessfulMlUpdate && isPostSuccessRetryNoiseBlock(block)) {
        return false;
      }
      return true;
    })
    .join("\n\n");
}

/**
 * Build the assistant message passed to the default renderer with a tool-trace suffix.
 *
 * @param message Raw assistant message from Copilot runtime.
 * @returns Message with transport JSON stripped and optional `Tools used:` suffix.
 */
export function buildRenderedAssistantMessage(
  message: AssistantMessageProps["message"]
): AssistantMessageProps["message"] {
  if (!message || typeof message.content !== "string") return message;

  const displayMessage = stripStandaloneToolResultBlocks(
    extractCopilotDisplayMessage(message.content)
  );
  const cleanedDisplayMessage = stripPostSuccessRetryNoise(displayMessage);
  const toolNames = [
    ...resolveToolNamesFromMessage(message),
    ...resolveToolNamesFromContent(message.content),
  ];
  const uniqueToolNames = Array.from(new Set(toolNames));
  const content = uniqueToolNames.length
    ? `${cleanedDisplayMessage}\n\nTools used: ${uniqueToolNames.join(", ")}`
    : cleanedDisplayMessage;
  return { ...message, content };
}

/**
 * Custom assistant renderer that hides transport JSON and renders only user-facing text.
 */
export default function CopilotAssistantMessage({
  message,
  ...props
}: AssistantMessageProps) {
  if (DEBUG_COPILOT && message && typeof message.content === "string") {
    const parsed = parseCopilotAssistantPayload(message.content);
    console.log("[agui-debug] assistant_message.received", {
      id: message.id,
      role: message.role,
      status: message.status,
      rawContentPreview: message.content.slice(0, 500),
      parsedPayload: parsed,
    });
  }

  const nextMessage = buildRenderedAssistantMessage(message);

  return <DefaultAssistantMessage {...props} message={nextMessage} />;
}
