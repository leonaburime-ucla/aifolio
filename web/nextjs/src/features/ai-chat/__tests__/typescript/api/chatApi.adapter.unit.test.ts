import { describe, expect, it, vi } from "vitest";

const {
  sendChatMessageMock,
  sendChatMessageDirectMock,
  fetchChatModelsMock,
} = vi.hoisted(() => ({
  sendChatMessageMock: vi.fn(async () => null),
  sendChatMessageDirectMock: vi.fn(async () => null),
  fetchChatModelsMock: vi.fn(async () => null),
}));

vi.mock("@/features/ai-chat/typescript/api/chatApi", () => ({
  sendChatMessage: sendChatMessageMock,
  sendChatMessageDirect: sendChatMessageDirectMock,
  fetchChatModels: fetchChatModelsMock,
}));

import { createChatApiAdapter } from "@/features/ai-chat/typescript/api/chatApi.adapter";

describe("chatApi.adapter", () => {
  it("uses research transport by default for research mode", () => {
    const adapter = createChatApiAdapter({ mode: "research" });

    expect(adapter.sendMessage).toBe(sendChatMessageMock);
    expect(adapter.fetchModels).toBe(fetchChatModelsMock);
  });

  it("uses direct transport for direct mode", () => {
    const adapter = createChatApiAdapter({ mode: "direct" });
    expect(adapter.sendMessage).toBe(sendChatMessageDirectMock);
  });

  it("honors explicit dependency overrides", () => {
    const sendResearchMessage = vi.fn(async () => null);
    const sendDirectMessage = vi.fn(async () => null);
    const fetchModels = vi.fn(async () => null);

    const adapter = createChatApiAdapter(
      { mode: "direct" },
      { sendResearchMessage, sendDirectMessage, fetchModels }
    );

    expect(adapter.sendMessage).toBe(sendDirectMessage);
    expect(adapter.fetchModels).toBe(fetchModels);
  });
});
