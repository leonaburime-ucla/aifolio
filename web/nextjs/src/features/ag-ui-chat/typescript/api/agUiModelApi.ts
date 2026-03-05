import { getAiApiBaseUrl } from "@/core/config/aiApi";
import type { AgUiModelOption } from "@/features/ag-ui-chat/__types__/typescript/react/state/agUiModel.types";

type ModelsResponse = {
  status?: string;
  currentModel?: string;
  models?: Array<{ id: string; label: string }>;
};

export type FetchAgUiModelsResult = {
  currentModel: string | null;
  models: AgUiModelOption[];
};

export async function fetchAgUiModels(timeoutMs = 5000): Promise<FetchAgUiModelsResult | null> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${getAiApiBaseUrl()}/llm/gemini-models`, {
      signal: controller.signal,
    });
    if (!response.ok) return null;

    const payload = (await response.json()) as ModelsResponse;
    if (payload.status !== "ok" || !Array.isArray(payload.models)) {
      return null;
    }

    return {
      currentModel: payload.currentModel ?? null,
      models: payload.models.map((model) => ({ id: model.id, label: model.label })),
    };
  } catch {
    return null;
  } finally {
    clearTimeout(timeoutId);
  }
}
