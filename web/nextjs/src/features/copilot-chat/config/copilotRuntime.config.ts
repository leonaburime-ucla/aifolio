import type { CopilotAgentName } from "@/features/copilot-chat/types/copilotChat.types";

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
  return {
    runtimeUrl: "/api/copilotkit",
    agent: "agentic-research" as const,
    backendBaseUrl:
      process.env.AG_UI_BASE_URL ||
      process.env.NEXT_PUBLIC_AI_API_URL ||
      "http://127.0.0.1:8000",
    backendAguiPath: "/agui",
  };
}

