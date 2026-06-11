import type { ChatModelOption } from "@aifolio/contracts/entities/chat";

export type ChatApi = {
  fetchModels: () => Promise<{ models: ChatModelOption[]; currentModel: string | null }>;
};

export function createChatApi({ baseUrl }: { baseUrl: string }): ChatApi {
  async function fetchModels() {
    const res = await fetch(`${baseUrl}/llm/gemini-models`);
    if (!res.ok) throw new Error("Failed to load models.");
    const data = (await res.json()) as {
      status: string;
      currentModel?: string;
      models?: ChatModelOption[];
    };
    if (data.status !== "ok" || !data.models) {
      throw new Error("Invalid models response.");
    }
    return {
      models: data.models,
      currentModel: data.currentModel ?? data.models[0]?.id ?? null,
    };
  }

  return { fetchModels };
}
