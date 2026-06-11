// Stays in Next.js app: wires Next.js runtime env (process.env.AG_UI_BASE_URL),
// CopilotKit agent name, and app route paths — all deployment-specific.
import type { CopilotAgentName } from "@aifolio/contracts/entities/ag-ui";
import { getAiApiBaseUrl } from "@/core/config/aiApi";

/**
 * Client-safe config used by React Copilot provider.
 * This file must not import server-only runtime packages.
 *
 * @returns Client runtime URL and default Copilot agent name.
 * @complexity O(1) time and space.
 * @overallScore 100
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
 *
 * @returns Server runtime URL, agent name, backend base URL, and AG-UI path.
 * @complexity O(1) time and space.
 * @overallScore 100
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
