import type { ChartSpec } from "@/features/ai-chat/__types__/typescript/chart.types";

export type CopilotAssistantPayload = {
  type?: "TextMessage";
  message: string;
  chartSpec: ChartSpec | ChartSpec[] | null;
};
