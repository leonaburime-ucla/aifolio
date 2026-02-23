import type { CopilotAgentName } from "@/features/copilot-chat/types/copilotChat.types";
import { getAiApiBaseUrl } from "@/core/config/aiApi";

/**
 * Client-safe config used by React Copilot provider.
 * This file must not import server-only runtime packages.
 */
export function getCopilotClientConfig(): {
  runtimeUrl: string;
  agent: CopilotAgentName;
} {
  return {
    runtimeUrl: "/api/copilotkit",
    agent: "agentic-research",
  };
}

/**
 * Server-side config used by Next API route adapter.
 */
export function getCopilotServerConfig() {
  const aiApiBaseUrl = getAiApiBaseUrl();
  return {
    runtimeUrl: "/api/copilotkit",
    agent: "agentic-research" as const,
    backendBaseUrl: process.env.AG_UI_BASE_URL || aiApiBaseUrl,
    backendAguiPath: "/agui",
  };
}
