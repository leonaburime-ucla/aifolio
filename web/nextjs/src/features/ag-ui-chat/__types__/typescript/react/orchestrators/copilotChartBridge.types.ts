import type { ChartSpec } from "@/features/ai-chat/__types__/typescript/chart.types";

export type MessageInfo = {
  messageId: string;
  messageType: string;
  isTextLike: boolean;
  messageRole: string;
  messageStatus: string;
  messageContent: string;
};

export type ProcessMessageResult =
  | { status: "skipped"; reason: string }
  | { status: "waiting"; reason: string }
  | { status: "no_chart_spec" }
  | { status: "charts_added"; count: number; ids: string[]; types: string[] };

export type ProcessMessageForChartSpecResult = {
  result: ProcessMessageResult;
  chartSpecs: ChartSpec[] | null;
};

export type CopilotChartBridgeOrchestrator = {
  processMessage: (messageInfo: MessageInfo) => void;
};
