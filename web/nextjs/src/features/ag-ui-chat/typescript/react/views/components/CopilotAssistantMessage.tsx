"use client";
import type { AssistantMessageProps } from "@copilotkit/react-ui";
import { AssistantMessage as DefaultAssistantMessage } from "@copilotkit/react-ui";
import {
  extractCopilotDisplayMessage,
  parseCopilotAssistantPayload,
} from "@/features/ag-ui-chat/typescript/logic/copilotAssistantPayload.util";

const DEBUG_COPILOT =
  process.env.NEXT_PUBLIC_COPILOT_DEBUG !== "0" &&
  process.env.COPILOT_DEBUG !== "0";

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

  const nextMessage =
    message && typeof message.content === "string"
      ? { ...message, content: extractCopilotDisplayMessage(message.content) }
      : message;

  return <DefaultAssistantMessage {...props} message={nextMessage} />;
}
