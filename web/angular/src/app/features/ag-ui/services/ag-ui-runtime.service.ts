import { inject, Injectable } from '@angular/core';
import { AgUiToolRegistryService } from './ag-ui-tool-registry.service';

export type AgUiRunResult = {
  message: string;
  toolCallsExecuted: { name: string; args: Record<string, unknown>; result: string }[];
};

type AgUiContext = { description: string; value: string };
type AgUiTool = { name: string; description: string; parameters: Record<string, unknown> };
type AgUiMessage = { id: string; role: string; content?: string; tool_call_id?: string };

export type AgUiRunInput = {
  messages: AgUiMessage[];
  context: AgUiContext[];
  userMessage: string;
};

@Injectable({ providedIn: 'root' })
export class AgUiRuntimeService {
  private readonly registry = inject(AgUiToolRegistryService);
  private abortController: AbortController | null = null;

  abort(): void {
    this.abortController?.abort();
  }

  async run(input: AgUiRunInput, baseUrl: string): Promise<AgUiRunResult> {
    this.abortController?.abort();
    this.abortController = new AbortController();

    const tools = this.buildToolDefinitions();
    const messages = this.buildMessages(input);
    const result = await this.executeRunLoop(messages, tools, input.context, baseUrl);
    return result;
  }

  private buildToolDefinitions(): AgUiTool[] {
    return this.registry.getRegisteredToolNames().map((name) => ({
      name,
      description: '',
      parameters: {},
    }));
  }

  private buildMessages(input: AgUiRunInput): AgUiMessage[] {
    return input.messages;
  }

  private async executeRunLoop(
    messages: AgUiMessage[],
    tools: AgUiTool[],
    context: AgUiContext[],
    baseUrl: string
  ): Promise<AgUiRunResult> {
    const toolCallsExecuted: AgUiRunResult['toolCallsExecuted'] = [];
    let currentMessages = [...messages];
    let iteration = 0;
    const maxIterations = 10;

    while (iteration < maxIterations) {
      iteration++;
      const { textContent, toolCalls } = await this.streamRun(currentMessages, tools, context, baseUrl);

      if (toolCalls.length === 0) {
        return { message: textContent, toolCallsExecuted };
      }

      for (const tc of toolCalls) {
        const result = await this.registry.execute(tc.name, tc.args);
        toolCallsExecuted.push({ name: tc.name, args: tc.args, result });

        currentMessages = [
          ...currentMessages,
          {
            id: `assistant_${iteration}`,
            role: 'assistant',
            content: textContent || undefined,
          },
          {
            id: `tool_result_${tc.id}`,
            role: 'tool',
            content: result,
            tool_call_id: tc.id,
          },
        ];
      }
    }

    return { message: 'Max tool iterations reached.', toolCallsExecuted };
  }

  private async streamRun(
    messages: AgUiMessage[],
    tools: AgUiTool[],
    context: AgUiContext[],
    baseUrl: string
  ): Promise<{ textContent: string; toolCalls: { id: string; name: string; args: Record<string, unknown> }[] }> {
    const threadId = this.generateId();
    const runId = this.generateId();

    const payload = {
      thread_id: threadId,
      run_id: runId,
      state: null,
      messages,
      tools,
      context,
      forwarded_props: null,
    };

    const response = await fetch(`${baseUrl}/agui`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: this.abortController?.signal,
    });

    if (!response.ok) {
      throw new Error(`AG-UI stream failed: ${response.status} ${response.statusText}`);
    }

    return this.parseSSEStream(response);
  }

  private async parseSSEStream(response: Response): Promise<{
    textContent: string;
    toolCalls: { id: string; name: string; args: Record<string, unknown> }[];
  }> {
    const text = await response.text();
    const lines = text.split('\n');

    let textContent = '';
    const toolCalls: { id: string; name: string; args: Record<string, unknown> }[] = [];
    const pendingToolCalls = new Map<string, { id: string; name: string; argsDelta: string }>();

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const raw = line.slice(6).trim();
      if (!raw) continue;

      let event: Record<string, unknown>;
      try {
        event = JSON.parse(raw);
      } catch {
        continue;
      }

      const type = event['type'] as string;

      switch (type) {
        case 'TOOL_CALL_START': {
          const id = event['toolCallId'] as string;
          const name = event['toolCallName'] as string;
          pendingToolCalls.set(id, { id, name, argsDelta: '' });
          break;
        }
        case 'TOOL_CALL_ARGS': {
          const id = event['toolCallId'] as string;
          const delta = event['delta'] as string;
          const pending = pendingToolCalls.get(id);
          if (pending) pending.argsDelta += delta;
          break;
        }
        case 'TOOL_CALL_END': {
          const id = event['toolCallId'] as string;
          const pending = pendingToolCalls.get(id);
          if (pending) {
            let args: Record<string, unknown> = {};
            try {
              args = JSON.parse(pending.argsDelta);
            } catch { /* empty */ }
            toolCalls.push({ id: pending.id, name: pending.name, args });
            pendingToolCalls.delete(id);
          }
          break;
        }
        case 'TEXT_MESSAGE_CONTENT': {
          textContent += event['delta'] as string;
          break;
        }
        case 'RUN_ERROR': {
          throw new Error((event['message'] as string) || 'AG-UI run error');
        }
      }
    }

    return { textContent, toolCalls };
  }

  private generateId(): string {
    return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
  }
}
