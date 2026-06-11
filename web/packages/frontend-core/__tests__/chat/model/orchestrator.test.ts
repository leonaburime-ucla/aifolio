import { describe, expect, it, vi } from "vitest";
import {
  createChatApiDeps,
  createChatDeps,
} from "../../../src/chat/model/orchestrator";

describe("createChatApiDeps", () => {
  it("returns object with sendMessage and fetchModels", () => {
    const sendMessage = vi.fn();
    const fetchModels = vi.fn();
    const result = createChatApiDeps({ sendMessage, fetchModels });
    expect(result.sendMessage).toBe(sendMessage);
    expect(result.fetchModels).toBe(fetchModels);
  });
});

describe("createChatDeps", () => {
  it("assembles full ChatDeps from state, actions, api, and logic", () => {
    const state = { messages: [], isSending: false } as any;
    const actions = { addMessage: vi.fn() } as any;
    const api = { sendMessage: vi.fn(), fetchModels: vi.fn() };
    const logic = { buildSubmission: vi.fn() } as any;

    const result = createChatDeps({ state, actions, api, logic });
    expect(result.state).toBe(state);
    expect(result.actions).toBe(actions);
    expect(result.api).toBe(api);
    expect(result.logic).toBe(logic);
  });
});
