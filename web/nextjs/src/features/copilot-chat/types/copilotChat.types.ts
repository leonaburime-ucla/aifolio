/**
 * Stable agent identifier configured in Copilot runtime and used by the UI provider.
 */
export type CopilotAgentName = "agentic-research";

/**
 * Runtime wiring settings shared between:
 * - Next.js API route (`/api/copilotkit`)
 * - React Copilot provider (`runtimeUrl`, `agent`)
 */
export type CopilotRuntimeConfig = {
  runtimeUrl: string;
  agent: CopilotAgentName;
  backendBaseUrl: string;
  backendAguiPath: string;
};

