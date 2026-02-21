"use client";

import { useEffect, useRef } from "react";
import { useCopilotMessagesContext } from "@copilotkit/react-core";
import { useChartStore } from "@/features/recharts/state/zustand/chartStore";
import { parseCopilotAssistantPayload } from "@/features/copilot-chat/utils/copilotAssistantPayload.util";

/**
 * Watches Copilot assistant messages and pushes validated chart specs into Zustand.
 *
 * Notes:
 * - Processes each assistant message once by message ID.
 * - Uses narrow effect dependencies to avoid broad `useEffect` loops.
 */
export default function CopilotChartBridge() {
  const { messages } = useCopilotMessagesContext();
  const addChartSpec = useChartStore((state) => state.addChartSpec);
  const processedMessageIdsRef = useRef<Set<string>>(new Set());

  const lastMessage = messages[messages.length - 1];
  const lastMessageId = lastMessage?.id ?? "";
  const lastMessageType = lastMessage?.type ?? "";
  const lastMessageRole =
    "role" in (lastMessage ?? {}) ? String((lastMessage as { role?: unknown }).role ?? "") : "";
  const lastMessageStatus =
    "status" in (lastMessage ?? {}) ? String((lastMessage as { status?: unknown }).status ?? "") : "";
  const lastMessageContent =
    lastMessageType === "TextMessage" && "content" in (lastMessage ?? {})
      ? String((lastMessage as { content?: unknown }).content ?? "")
      : "";

  useEffect(() => {
    if (!lastMessageId || lastMessageType !== "TextMessage") {
      console.log("[copilot-chart-bridge] skip.non_text_or_missing_id", {
        lastMessageId,
        lastMessageType,
      });
      return;
    }
    if (processedMessageIdsRef.current.has(lastMessageId)) {
      console.log("[copilot-chart-bridge] skip.already_processed", { lastMessageId });
      return;
    }

    if (lastMessageRole !== "assistant") {
      console.log("[copilot-chart-bridge] skip.non_assistant", {
        lastMessageId,
        lastMessageRole,
      });
      return;
    }

    const payload = parseCopilotAssistantPayload(lastMessageContent);
    console.log("[copilot-chart-bridge] message.received", {
      lastMessageId,
      status: lastMessageStatus,
      rawPreview: lastMessageContent.slice(0, 300),
      parsed: payload,
    });
    if (!payload) {
      // Keep waiting for full streamed content; do not mark as processed on parse failure.
      console.log("[copilot-chart-bridge] waiting_for_parseable_payload", { lastMessageId });
      return;
    }

    if (!payload.chartSpec) {
      processedMessageIdsRef.current.add(lastMessageId);
      console.log("[copilot-chart-bridge] no_chart_spec", { lastMessageId });
      return;
    }

    if (Array.isArray(payload.chartSpec)) {
      payload.chartSpec.forEach((spec) => addChartSpec(spec));
      console.log("[copilot-chart-bridge] chart_specs_added", {
        lastMessageId,
        count: payload.chartSpec.length,
        ids: payload.chartSpec.map((spec) => spec.id),
        types: payload.chartSpec.map((spec) => spec.type),
      });
    } else {
      addChartSpec(payload.chartSpec);
      console.log("[copilot-chart-bridge] chart_spec_added", {
        lastMessageId,
        id: payload.chartSpec.id,
        type: payload.chartSpec.type,
      });
    }

    processedMessageIdsRef.current.add(lastMessageId);
  }, [addChartSpec, lastMessageContent, lastMessageId, lastMessageRole, lastMessageStatus, lastMessageType]);

  return null;
}
