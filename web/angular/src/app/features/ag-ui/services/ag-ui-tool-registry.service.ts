import { Injectable } from '@angular/core';

export type ToolHandler = (args: Record<string, unknown>) => unknown | Promise<unknown>;

@Injectable({ providedIn: 'root' })
export class AgUiToolRegistryService {
  private readonly handlers = new Map<string, ToolHandler>();

  register(name: string, handler: ToolHandler): void {
    this.handlers.set(name, handler);
  }

  unregister(name: string): void {
    this.handlers.delete(name);
  }

  has(name: string): boolean {
    return this.handlers.has(name);
  }

  async execute(name: string, args: Record<string, unknown>): Promise<string> {
    const handler = this.handlers.get(name);
    if (!handler) {
      return JSON.stringify({ status: 'error', code: 'UNKNOWN_TOOL', tool: name });
    }
    try {
      const result = await handler(args);
      return typeof result === 'string' ? result : JSON.stringify(result ?? { status: 'ok' });
    } catch (err) {
      return JSON.stringify({ status: 'error', code: 'TOOL_EXECUTION_FAILED', message: String(err) });
    }
  }

  getRegisteredToolNames(): string[] {
    return Array.from(this.handlers.keys());
  }
}
