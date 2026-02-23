"use client";

import { CopilotKit } from "@copilotkit/react-core";
import "@copilotkit/react-ui/styles.css";
import { getCopilotClientConfig } from "@/features/copilot-chat/config/copilotRuntime.config";
import { CopilotEffectsProvider } from "@/features/copilot-chat/views/providers/CopilotEffectsProvider";

type CopilotChatProviderProps = {
  children: React.ReactNode;
};

/**
 * Feature-local Copilot provider.
 *
 * This keeps Copilot wiring inside the `copilot-chat` vertical slice so
 * the feature can be reused or moved without touching generic core providers.
 *
 * Includes CopilotEffectsProvider which consolidates all "invisible" side-effects:
 * - Frontend tool registrations (useCopilotAction)
 * - Chart bridge (message to store sync)
 * - Message persistence (localStorage)
 */
export default function CopilotChatProvider({
  children,
}: CopilotChatProviderProps) {
  const config = getCopilotClientConfig();
  return (
    <CopilotKit runtimeUrl={config.runtimeUrl} agent={config.agent}>
      <CopilotEffectsProvider>
        {children}
      </CopilotEffectsProvider>
    </CopilotKit>
  );
}
