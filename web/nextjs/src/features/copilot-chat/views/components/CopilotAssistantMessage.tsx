"use client";
import type { AssistantMessageProps } from "@copilotkit/react-ui";
import { AssistantMessage as DefaultAssistantMessage } from "@copilotkit/react-ui";
import { extractCopilotDisplayMessage } from "@/features/copilot-chat/utils/copilotAssistantPayload.util";

/**
 * Custom assistant renderer that hides transport JSON and renders only user-facing text.
 */
export default function CopilotAssistantMessage({
  message,
  ...props
}: AssistantMessageProps) {
  const nextMessage =
    message && typeof message.content === "string"
      ? { ...message, content: extractCopilotDisplayMessage(message.content) }
      : message;

  return <DefaultAssistantMessage {...props} message={nextMessage} />;
}
