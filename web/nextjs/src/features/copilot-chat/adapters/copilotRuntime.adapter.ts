import "server-only";

import {
  CopilotRuntime,
  EmptyAdapter,
  copilotRuntimeNextJSAppRouterEndpoint,
} from "@copilotkit/runtime";
import { LangGraphHttpAgent } from "@copilotkit/runtime/langgraph";
import { getCopilotServerConfig } from "@/features/copilot-chat/config/copilotRuntime.config";

/**
 * Create the Next.js App Router request handler for `/api/copilotkit`.
 *
 * Notes:
 * - Uses `LangGraphHttpAgent` to forward to the backend AG-UI stream.
 * - Uses `EmptyAdapter` as a runtime compatibility workaround for
 *   `@copilotkit/runtime@1.51.3` which expects `serviceAdapter`.
 */
export function createCopilotAppRouterHandler() {
  const config = getCopilotServerConfig();
  const runtime = new CopilotRuntime({
    agents: {
      [config.agent]: new LangGraphHttpAgent({
        url: `${config.backendBaseUrl}${config.backendAguiPath}`,
      }),
    },
  });

  const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
    runtime,
    serviceAdapter: new EmptyAdapter(),
    endpoint: config.runtimeUrl,
  });

  return handleRequest;
}
