"use client";

/**
 * Purpose: Observe Copilot assistant messages and project valid chart specs into chart state.
 */
import { useCallback, useRef, useEffect } from "react";
import { useCopilotChatInternal } from "@copilotkit/react-core";
import { useCopilotChartActionsAdapter } from "@/features/recharts/typescript/react/ai/state/adapters/chartActions.adapter";
import { useAgenticResearchChartActionsAdapter } from "@/features/agentic-research/typescript/react/state/adapters/chartActions.adapter";
import { useAgUiWorkspaceStateAdapter } from "@/features/ag-ui-chat/typescript/react/state/adapters/agUiWorkspaceState.adapter";
import { parseCopilotAssistantPayload } from "@/features/ag-ui-chat/typescript/logic/copilotAssistantPayload.util";
import type { ChartSpec } from "@/features/ai-chat/__types__/typescript/chart.types";
import type {
  CopilotChartBridgeOrchestrator,
  MessageInfo,
  ProcessMessageForChartSpecResult,
} from "@/features/ag-ui-chat/__types__/typescript/react/orchestrators/copilotChartBridge.types";

const DEBUG_COPILOT =
  process.env.NEXT_PUBLIC_COPILOT_DEBUG !== "0" &&
  process.env.COPILOT_DEBUG !== "0";

function debugLog(event: string, meta: Record<string, unknown>) {
  if (!DEBUG_COPILOT) return;
  console.log(event, meta);
}

/**
 * Pure function to process a copilot message and extract chart specs.
 * Returns the result of processing without side effects.
 */
export function processMessageForChartSpec(
  messageInfo: MessageInfo,
  isAlreadyProcessed: boolean
): ProcessMessageForChartSpecResult {
  const { messageId, messageType, isTextLike, messageRole, messageContent } = messageInfo;

  // Skip non-text or missing ID
  if (!messageId || !isTextLike) {
    debugLog("[copilot-chart-bridge] skip.non_text_or_missing_id", {
      lastMessageId: messageId,
      lastMessageType: messageType,
      isTextLike,
    });
    return {
      result: { status: "skipped", reason: "non_text_or_missing_id" },
      chartSpecs: null,
    };
  }

  // Skip already processed
  if (isAlreadyProcessed) {
    debugLog("[copilot-chart-bridge] skip.already_processed", { lastMessageId: messageId });
    return {
      result: { status: "skipped", reason: "already_processed" },
      chartSpecs: null,
    };
  }

  // Skip non-assistant messages
  if (messageRole !== "assistant") {
    debugLog("[copilot-chart-bridge] skip.non_assistant", {
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
  debugLog("[copilot-chart-bridge] message.received", {
    lastMessageId: messageId,
    status: messageInfo.messageStatus,
    rawPreview: messageContent.slice(0, 300),
    parsed: payload,
  });

  // Wait for parseable content (streaming might not be complete)
  if (!payload) {
    debugLog("[copilot-chart-bridge] waiting_for_parseable_payload", { lastMessageId: messageId });
    return {
      result: { status: "waiting", reason: "waiting_for_parseable_payload" },
      chartSpecs: null,
    };
  }

  // No chart spec in payload
  if (!payload.chartSpec) {
    debugLog("[copilot-chart-bridge] no_chart_spec", { lastMessageId: messageId });
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
  const { addChartSpec: addGlobalChartSpec } = useCopilotChartActionsAdapter();
  const { addChartSpec: addAgenticChartSpec } = useAgenticResearchChartActionsAdapter();
  const { activeTab } = useAgUiWorkspaceStateAdapter();

  // Use Copilot internal chat state as the single source of truth.
  const { messages } = useCopilotChatInternal();

  // Track processed message IDs
  const processedMessageIdsRef = useRef<Set<string>>(new Set());

  // Create stable handler reference
  const processMessage = useCallback(
    (messageInfo: MessageInfo) => {
      const isAlreadyProcessed = processedMessageIdsRef.current.has(messageInfo.messageId);

      const { result, chartSpecs } = processMessageForChartSpec(messageInfo, isAlreadyProcessed);
      debugLog("[agui-debug] chart_bridge.processed", {
        messageId: messageInfo.messageId,
        status: result.status,
        isAlreadyProcessed,
        contentLength: messageInfo.messageContent.length,
        hasChartSpecs: Array.isArray(chartSpecs) ? chartSpecs.length > 0 : false,
        chartCount: Array.isArray(chartSpecs) ? chartSpecs.length : 0,
        activeTab,
      });

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
        // Route chart injection to the active AG-UI workspace surface.
        chartSpecs.forEach((spec) => {
          if (activeTab === "agentic-research") {
            addAgenticChartSpec(spec);
            return;
          }
          addGlobalChartSpec(spec);
        });

        debugLog("[copilot-chart-bridge] chart_specs_added", {
          lastMessageId: messageInfo.messageId,
          count: result.count,
          ids: result.ids,
          types: result.types,
          targetStore: activeTab === "agentic-research" ? "agentic-research" : "global",
        });

        // Mark as processed
        processedMessageIdsRef.current.add(messageInfo.messageId);
      }
    },
    [activeTab, addAgenticChartSpec, addGlobalChartSpec]
  );

  useEffect(() => {
    if (!Array.isArray(messages) || messages.length === 0) {
      return;
    }

    const pending: MessageInfo[] = [];
    for (const message of messages as unknown[]) {
      const messageInfo = extractMessageInfo(message);
      if (!messageInfo) continue;
      if (!messageInfo.messageId) continue;
      if (processedMessageIdsRef.current.has(messageInfo.messageId)) continue;
      if (!messageInfo.isTextLike) continue;
      if (messageInfo.messageRole !== "assistant") continue;
      if (!messageInfo.messageContent.trim()) continue;
      pending.push(messageInfo);
    }

    if (!pending.length) return;
    pending.forEach((messageInfo) => processMessage(messageInfo));
  }, [messages, processMessage]);

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
  const rawContent = "content" in msg ? msg.content : "";
  const messageContent =
    typeof rawContent === "string"
      ? rawContent
      : Array.isArray(rawContent)
        ? rawContent
          .map((part) => {
            if (typeof part === "string") return part;
            if (!part || typeof part !== "object") return "";
            const record = part as Record<string, unknown>;
            if (typeof record.text === "string") return record.text;
            if (typeof record.content === "string") return record.content;
            return "";
          })
          .filter(Boolean)
          .join("\n")
        : "";
  const isTextLike = messageType === "TextMessage" || messageType === "" || messageContent.length > 0;

  if (!messageId) {
    return null;
  }

  return {
    messageId,
    messageType,
    isTextLike,
    messageRole,
    messageStatus,
    messageContent,
  };
}
