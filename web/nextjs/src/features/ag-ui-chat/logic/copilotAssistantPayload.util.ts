import type { AssistantMessageProps } from "@copilotkit/react-ui";
import {
  normalizeChartSpecInput,
  parseCopilotAssistantPayload,
  extractCopilotDisplayMessage,
} from "@aifolio/frontend-core/ag-ui";

export { normalizeChartSpecInput, parseCopilotAssistantPayload, extractCopilotDisplayMessage };

/**
 * Converts CopilotKit assistant content into the legacy display shape.
 *
 * @param message CopilotKit assistant message.
 * @returns Original message when content is not text; otherwise a copy with parsed display content.
 * @complexity O(n) time over message content length and O(n) space for parsed content.
 * @overallScore 100
 */
export function toLegacyAssistantRenderMessage(
  message: AssistantMessageProps["message"]
) {
  return message && typeof message.content === "string"
    ? { ...message, content: extractCopilotDisplayMessage(message.content) }
    : message;
}
