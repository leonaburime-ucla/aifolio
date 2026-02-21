import { createCopilotAppRouterHandler } from "@/features/copilot-chat/adapters/copilotRuntime.adapter";

/**
 * Next.js App Router endpoint consumed by Copilot React UI.
 * Handler creation is delegated to the copilot feature adapter.
 */
const handleRequest = createCopilotAppRouterHandler();

/**
 * Debug toggle:
 * - Set `NEXT_PUBLIC_COPILOT_DEBUG=1` or `COPILOT_DEBUG=1` in frontend env.
 * - Logs request/response metadata for `/api/copilotkit` bridge calls.
 */
const DEBUG_COPILOT =
  process.env.NEXT_PUBLIC_COPILOT_DEBUG === "1" ||
  process.env.COPILOT_DEBUG === "1";

export async function POST(req: Request): Promise<Response> {
  const startedAt = Date.now();
  if (DEBUG_COPILOT) {
    const bodyPreview = await req.clone().text().catch(() => "");
    console.log("[copilot-route] request", {
      method: req.method,
      url: req.url,
      bodyPreview: bodyPreview.slice(0, 500),
    });
  }

  const response = await handleRequest(req);

  if (DEBUG_COPILOT) {
    console.log("[copilot-route] response", {
      status: response.status,
      elapsedMs: Date.now() - startedAt,
    });
  }
  return response;
}
