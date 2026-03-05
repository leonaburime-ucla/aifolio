import type { AssistantMessageProps } from "@copilotkit/react-ui";
import type { CopilotAssistantMessageOrchestratorDeps } from "@/features/ag-ui-chat/__types__/typescript/react/orchestrators/copilotAssistantMessageOrchestrator.types";

export type CopilotAssistantMessageLegacyProps = AssistantMessageProps &
  CopilotAssistantMessageOrchestratorDeps;
