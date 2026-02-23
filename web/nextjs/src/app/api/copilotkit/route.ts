import { createCopilotAppRouterHandler } from "@/features/copilot-chat/adapters/copilotRuntime.adapter";

/**
 * Next.js App Router endpoint consumed by Copilot React UI.
 * Handler creation is delegated to the copilot feature adapter.
 */
const handleRequest = createCopilotAppRouterHandler();

export async function POST(req: Request): Promise<Response> {
  return handleRequest(req);
}
