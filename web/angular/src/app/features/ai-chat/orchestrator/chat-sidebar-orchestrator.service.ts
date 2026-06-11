import { Injectable, computed, inject, signal } from '@angular/core';
import { sendChatMessage, sendChatMessageDirect } from '@aifolio/frontend-core/chat';
import type { ChatHistoryDirection, ChatHistoryMessage, ChatMessage, ChatModelOption, ScreenFeedback } from '@aifolio/contracts/entities/chat';
import type { ChartSpec } from '@aifolio/contracts/entities/chart';
import { AI_API_BASE_URL } from '../../../core/config/ai-api.config';
import { ChartStoreService } from '../../../shared/state/chart-store.service';
import { ChatApiService } from '../api/chat-api.service';

type ChatMode = 'direct' | 'research';

@Injectable()
export class ChatSidebarOrchestrator {
  private readonly api = inject(ChatApiService);
  private readonly chartStore = inject(ChartStoreService);

  readonly baseUrl = signal(AI_API_BASE_URL);
  readonly mode = signal<ChatMode>('direct');
  readonly datasetId = signal<string | null>(null);
  readonly messages = signal<ChatMessage[]>([]);
  readonly inputValue = signal('');
  readonly isSending = signal(false);
  readonly modelOptions = signal<ChatModelOption[]>([]);
  readonly selectedModelId = signal<string | null>(null);
  readonly inputHistory = signal<string[]>([]);
  readonly historyCursor = signal<number | null>(null);
  readonly screenFeedback = signal<ScreenFeedback | null>(null);
  readonly hasInput = computed(() => this.inputValue().trim().length > 0);

  configure(input: { baseUrl?: string; mode?: ChatMode; datasetId?: string | null }): void {
    if (input.baseUrl) this.baseUrl.set(input.baseUrl);
    if (input.mode) this.mode.set(input.mode);
    this.datasetId.set(input.datasetId ?? null);
  }

  async loadModels(): Promise<void> {
    try {
      const { models, currentModel } = await this.api.fetchModels(this.baseUrl());
      this.modelOptions.set(models);
      this.selectedModelId.set(currentModel);
    } catch {
      // Non-critical. The backend can still accept its default model.
    }
  }

  async submit(): Promise<void> {
    const text = this.inputValue().trim();
    if (!text || this.isSending()) return;

    this.screenFeedback.set(null);
    const userMsg: ChatMessage = { id: this.generateId(), role: 'user', content: text, createdAt: Date.now() };
    this.messages.update((messages) => [...messages, userMsg]);
    this.inputHistory.update((history) => [...history, text]);
    this.historyCursor.set(null);
    this.inputValue.set('');
    this.isSending.set(true);

    const history: ChatHistoryMessage[] = this.messages()
      .slice(-10)
      .map((message) => ({ role: message.role, content: message.content }));

    try {
      const sender = this.mode() === 'research' ? sendChatMessage : sendChatMessageDirect;
      const result = await sender(
        {
          value: text,
          model: this.selectedModelId(),
          history,
          attachments: undefined,
        },
        {
          runtimeDeps: { resolveBaseUrl: () => this.baseUrl() },
          ...(this.datasetId() ? { datasetId: this.datasetId() } : {}),
        }
      );

      if (!result) {
        this.screenFeedback.set({ kind: 'error', code: 'CHAT_EMPTY_RESPONSE', message: 'No response from the backend.' });
        return;
      }

      this.messages.update((messages) => [
        ...messages,
        { id: this.generateId(), role: 'assistant', content: result.message, createdAt: Date.now(), chartSpec: result.chartSpec as ChartSpec | null },
      ]);

      const specs = Array.isArray(result.chartSpec) ? result.chartSpec : result.chartSpec ? [result.chartSpec] : [];
      for (const spec of specs) {
        if (spec?.data) this.addChartSpec(spec as ChartSpec);
      }
    } catch (err) {
      this.screenFeedback.set({
        kind: 'error',
        code: 'CHAT_REQUEST_FAILED',
        message: err instanceof Error ? err.message : 'Request failed.',
        retryable: true,
      });
    } finally {
      this.isSending.set(false);
    }
  }

  handleHistory(direction: ChatHistoryDirection): void {
    const history = this.inputHistory();
    if (history.length === 0) return;

    if (direction === 'up') {
      const next = this.historyCursor() === null ? history.length - 1 : Math.max(0, this.historyCursor()! - 1);
      this.historyCursor.set(next);
      this.inputValue.set(history[next]);
      return;
    }

    if (this.historyCursor() === null) return;
    const next = this.historyCursor()! + 1;
    if (next >= history.length) {
      this.historyCursor.set(null);
      this.inputValue.set('');
      return;
    }
    this.historyCursor.set(next);
    this.inputValue.set(history[next]);
  }

  private addChartSpec(raw: ChartSpec): void {
    this.chartStore.addChartSpec({
      id: raw.id || Date.now().toString(36),
      title: raw.title || 'Chart',
      type: raw.type || 'line',
      xKey: raw.xKey || 'x',
      yKeys: raw.yKeys ?? [],
      xLabel: raw.xLabel,
      yLabel: raw.yLabel,
      data: raw.data,
      description: raw.description,
      meta: raw.meta,
    });
  }

  private generateId(): string {
    return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
  }
}
