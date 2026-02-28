"use client";

import { useCallback, useRef, useEffect } from "react";
import { useCopilotMessagesContext } from "@copilotkit/react-core";
import { useRechartsChartActionsAdapter } from "@/features/recharts/state/adapters/chartActions.adapter";
import { parseCopilotAssistantPayload } from "@/features/copilot-chat/utils/copilotAssistantPayload.util";
import type { ChartSpec } from "@/features/ai/types/chart.types";

/**
 * Processed message info used for determining if processing should occur.
 */
type MessageInfo = {
  messageId: string;
  messageType: string;
  messageRole: string;
  messageStatus: string;
  messageContent: string;
};

/**
 * Result of processing a copilot message.
 */
type ProcessMessageResult =
  | { status: "skipped"; reason: string }
  | { status: "waiting"; reason: string }
  | { status: "no_chart_spec" }
  | { status: "charts_added"; count: number; ids: string[]; types: string[] };

/**
 * Pure function to process a copilot message and extract chart specs.
 * Returns the result of processing without side effects.
 */
export function processMessageForChartSpec(
  messageInfo: MessageInfo,
  isAlreadyProcessed: boolean
): { result: ProcessMessageResult; chartSpecs: ChartSpec[] | null } {
  const { messageId, messageType, messageRole, messageContent } = messageInfo;

  // Skip non-text or missing ID
  if (!messageId || messageType !== "TextMessage") {
    console.log("[copilot-chart-bridge] skip.non_text_or_missing_id", {
      lastMessageId: messageId,
      lastMessageType: messageType,
    });
    return {
      result: { status: "skipped", reason: "non_text_or_missing_id" },
      chartSpecs: null,
    };
  }

  // Skip already processed
  if (isAlreadyProcessed) {
    console.log("[copilot-chart-bridge] skip.already_processed", { lastMessageId: messageId });
    return {
      result: { status: "skipped", reason: "already_processed" },
      chartSpecs: null,
    };
  }

  // Skip non-assistant messages
  if (messageRole !== "assistant") {
    console.log("[copilot-chart-bridge] skip.non_assistant", {
      lastMessageId: messageId,
      lastMessageRole: messageRole,
    });
    return {
      result: { status: "skipped", reason: "non_assistant" },
      chartSpecs: null,
    };
  }

  // Parse the message content
  const payload = parseCopilotAssistantPayload(messageContent);
  console.log("[copilot-chart-bridge] message.received", {
    lastMessageId: messageId,
    status: messageInfo.messageStatus,
    rawPreview: messageContent.slice(0, 300),
    parsed: payload,
  });

  // Wait for parseable content (streaming might not be complete)
  if (!payload) {
    console.log("[copilot-chart-bridge] waiting_for_parseable_payload", { lastMessageId: messageId });
    return {
      result: { status: "waiting", reason: "waiting_for_parseable_payload" },
      chartSpecs: null,
    };
  }

  // No chart spec in payload
  if (!payload.chartSpec) {
    console.log("[copilot-chart-bridge] no_chart_spec", { lastMessageId: messageId });
    return {
      result: { status: "no_chart_spec" },
      chartSpecs: null,
    };
  }

  // Extract chart specs
  const specs = Array.isArray(payload.chartSpec) ? payload.chartSpec : [payload.chartSpec];

  return {
    result: {
      status: "charts_added",
      count: specs.length,
      ids: specs.map((spec) => spec.id),
      types: specs.map((spec) => spec.type),
    },
    chartSpecs: specs,
  };
}

/**
 * Return type for the CopilotChartBridge orchestrator hook.
 */
export type CopilotChartBridgeOrchestrator = {
  /**
   * Process a copilot message and add any chart specs to the store.
   * Call this from useEffect when message changes.
   */
  processMessage: (messageInfo: MessageInfo) => void;
};

/**
 * Orchestrator hook for CopilotChartBridge.
 *
 * This hook is self-contained and watches Copilot assistant messages,
 * pushing validated chart specs into the store.
 *
 * Previously, the message watching logic was in a separate component (CopilotChartBridge.tsx).
 * Now it is consolidated here, eliminating the need for "invisible" null-returning components.
 *
 * Notes:
 * - Processes each assistant message once by message ID.
 * - Uses narrow effect dependencies to avoid broad `useEffect` loops.
 */
export function useCopilotChartBridgeOrchestrator(): CopilotChartBridgeOrchestrator {
  // Import adapters internally - view should not import these directly
  const { addChartSpec } = useRechartsChartActionsAdapter();

  // Get messages from CopilotKit context
  const { messages } = useCopilotMessagesContext();

  // Track processed message IDs
  const processedMessageIdsRef = useRef<Set<string>>(new Set());

  // Create stable handler reference
  const processMessage = useCallback(
    (messageInfo: MessageInfo) => {
      const isAlreadyProcessed = processedMessageIdsRef.current.has(messageInfo.messageId);

      const { result, chartSpecs } = processMessageForChartSpec(messageInfo, isAlreadyProcessed);

      // Handle results
      if (result.status === "skipped" || result.status === "waiting") {
        // Don't mark as processed - waiting or skipped
        return;
      }

      if (result.status === "no_chart_spec") {
        // Mark as processed but no charts to add
        processedMessageIdsRef.current.add(messageInfo.messageId);
        return;
      }

      if (result.status === "charts_added" && chartSpecs) {
        // Add each chart spec to the store
        chartSpecs.forEach((spec) => addChartSpec(spec));

        console.log("[copilot-chart-bridge] chart_specs_added", {
          lastMessageId: messageInfo.messageId,
          count: result.count,
          ids: result.ids,
          types: result.types,
        });

        // Mark as processed
        processedMessageIdsRef.current.add(messageInfo.messageId);
      }
    },
    [addChartSpec]
  );

  // Self-contained effect: watch messages and process them
  const lastMessage = messages[messages.length - 1];
  const messageInfo = extractMessageInfo(lastMessage);

  // Extract stable primitive values for effect dependencies
  const messageId = messageInfo?.messageId ?? "";
  const messageType = messageInfo?.messageType ?? "";
  const messageRole = messageInfo?.messageRole ?? "";
  const messageStatus = messageInfo?.messageStatus ?? "";
  const messageContent = messageInfo?.messageContent ?? "";

  useEffect(() => {
    // Skip if no valid message ID (i.e., messageInfo would be null)
    if (!messageId) {
      return;
    }

    // Reconstruct messageInfo from primitives inside effect to satisfy exhaustive-deps
    processMessage({
      messageId,
      messageType,
      messageRole,
      messageStatus,
      messageContent,
    });
  }, [processMessage, messageId, messageType, messageRole, messageStatus, messageContent]);

  return {
    processMessage,
  };
}

/**
 * Extract message info from a copilot message object.
 * Helper function to normalize message data for processing.
 */
export function extractMessageInfo(message: unknown): MessageInfo | null {
  if (!message || typeof message !== "object") {
    return null;
  }

  const msg = message as Record<string, unknown>;

  const messageId = typeof msg.id === "string" ? msg.id : "";
  const messageType = typeof msg.type === "string" ? msg.type : "";
  const messageRole = "role" in msg ? String(msg.role ?? "") : "";
  const messageStatus = "status" in msg ? String(msg.status ?? "") : "";
  const messageContent =
    messageType === "TextMessage" && "content" in msg
      ? String(msg.content ?? "")
      : "";

  if (!messageId) {
    return null;
  }

  return {
    messageId,
    messageType,
    messageRole,
    messageStatus,
    messageContent,
  };
}
