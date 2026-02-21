"use client";

import { CopilotKit } from "@copilotkit/react-core";
import "@copilotkit/react-ui/styles.css";
import { getCopilotClientConfig } from "@/features/copilot-chat/config/copilotRuntime.config";

type CopilotChatProviderProps = {
  children: React.ReactNode;
};

/**
 * Feature-local Copilot provider.
 *
 * This keeps Copilot wiring inside the `copilot-chat` vertical slice so
 * the feature can be reused or moved without touching generic core providers.
 */
export default function CopilotChatProvider({
  children,
}: CopilotChatProviderProps) {
  const config = getCopilotClientConfig();
  return (
    <CopilotKit runtimeUrl={config.runtimeUrl} agent={config.agent}>
      {children}
    </CopilotKit>
  );
}
