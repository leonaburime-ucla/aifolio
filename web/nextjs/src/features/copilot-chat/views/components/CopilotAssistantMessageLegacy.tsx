"use client";

import { useEffect } from "react";
import type { AssistantMessageProps } from "@copilotkit/react-ui";
import { AssistantMessage as DefaultAssistantMessage } from "@copilotkit/react-ui";
import {
  extractCopilotDisplayMessage,
  parseCopilotAssistantPayload,
} from "@/features/copilot-chat/utils/copilotAssistantPayload.util";
import { useChartStore } from "@/features/recharts/state/zustand/chartStore";

/**
 * Legacy assistant renderer for `/` AI Chat:
 * still parses assistant JSON payload and pushes `chartSpec` to chart store.
 */
export default function CopilotAssistantMessageLegacy({
  message,
  ...props
}: AssistantMessageProps) {
  const rawContent = message?.content || "";
  const addChartSpec = useChartStore((state) => state.addChartSpec);

  useEffect(() => {
    if (!rawContent) return;
    const payload = parseCopilotAssistantPayload(rawContent);
    if (!payload?.chartSpec) return;

    if (Array.isArray(payload.chartSpec)) {
      payload.chartSpec.forEach((spec) => addChartSpec(spec));
      console.log("[copilot-assistant-legacy] chart_specs_added", {
        count: payload.chartSpec.length,
        ids: payload.chartSpec.map((spec) => spec.id),
        types: payload.chartSpec.map((spec) => spec.type),
      });
      return;
    }

    addChartSpec(payload.chartSpec);
    console.log("[copilot-assistant-legacy] chart_spec_added", {
      id: payload.chartSpec.id,
      type: payload.chartSpec.type,
    });
  }, [addChartSpec, rawContent]);

  const nextMessage =
    message && typeof message.content === "string"
      ? { ...message, content: extractCopilotDisplayMessage(message.content) }
      : message;

  return <DefaultAssistantMessage {...props} message={nextMessage} />;
}
