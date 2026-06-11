import { Injectable } from '@angular/core';
import type { ChatModelOption } from '@aifolio/contracts/entities/chat';

@Injectable({ providedIn: 'root' })
export class ChatApiService {
  async fetchModels(baseUrl: string): Promise<{ models: ChatModelOption[]; currentModel: string | null }> {
    const res = await fetch(`${baseUrl}/llm/gemini-models`);
    if (!res.ok) throw new Error('Failed to load models.');
    const data = (await res.json()) as {
      status: string;
      currentModel?: string;
      models?: ChatModelOption[];
    };
    if (data.status !== 'ok' || !data.models) throw new Error('Invalid models response.');
    return {
      models: data.models,
      currentModel: data.currentModel ?? data.models[0]?.id ?? null,
    };
  }
}
