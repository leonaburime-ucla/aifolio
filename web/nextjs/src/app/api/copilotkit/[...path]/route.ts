import { createCopilotAppRouterHandler } from "@/features/ag-ui-chat/api/copilotRuntime.adapter";

const handleRequest = createCopilotAppRouterHandler();

export async function POST(req: Request): Promise<Response> {
  return handleRequest(req);
}
