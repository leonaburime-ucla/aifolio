import "server-only";

import {
  CopilotRuntime,
  EmptyAdapter,
  copilotRuntimeNextJSAppRouterEndpoint,
} from "@copilotkit/runtime";
import { LangGraphHttpAgent } from "@copilotkit/runtime/langgraph";
import { getCopilotServerConfig } from "@/features/ag-ui-chat/typescript/config/copilotRuntime.config";

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
  const runtimeParams = {
    agents: {
      // Bridge duplicate @ag-ui/client type instances across Copilot packages.
      [config.agent]: new LangGraphHttpAgent({
        url: `${config.backendBaseUrl}${config.backendAguiPath}`,
      }),
    },
  } as unknown as ConstructorParameters<typeof CopilotRuntime>[0];

  const runtime = new CopilotRuntime(runtimeParams);

  const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
    runtime,
    serviceAdapter: new EmptyAdapter(),
    endpoint: config.runtimeUrl,
  });

  return handleRequest;
}
