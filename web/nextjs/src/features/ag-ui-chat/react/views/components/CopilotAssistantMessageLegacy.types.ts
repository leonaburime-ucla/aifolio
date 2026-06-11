import type { AssistantMessageProps } from "@copilotkit/react-ui";
import type { CopilotAssistantMessageOrchestratorDeps } from "@/features/ag-ui-chat/react/orchestrators/copilotAssistantMessage.orchestrator.types";

export type CopilotAssistantMessageLegacyProps = AssistantMessageProps &
  CopilotAssistantMessageOrchestratorDeps;
