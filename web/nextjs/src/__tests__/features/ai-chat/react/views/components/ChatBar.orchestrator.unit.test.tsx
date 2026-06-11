import { describe, expect, it, vi } from "vitest";
import { render, fireEvent } from "@testing-library/react";
import ChatBar from "@/features/ai-chat/react/views/components/ChatBar";
import type { ChatIntegration } from "@aifolio/contracts/entities/chat";

function createMockOrchestrator(overrides: Partial<ChatIntegration> = {}): () => ChatIntegration {
  return () => ({
    value: "",
    showTooltip: false,
    attachments: [],
    messages: [],
    inputHistory: [],
    historyCursor: null,
    isSending: false,
    modelOptions: [],
    selectedModelId: null,
    isModelsLoading: false,
    screenFeedback: null,
    setShowTooltip: vi.fn(),
    setValue: vi.fn(),
    resetValue: vi.fn(),
    addAttachments: vi.fn(),
    clearAttachments: vi.fn(),
    removeAttachment: vi.fn(),
    submit: vi.fn(async () => {}),
    retryLastSubmission: vi.fn(async () => {}),
    handleHistory: vi.fn(),
    resetHistoryCursor: vi.fn(),
    setSelectedModelId: vi.fn(),
    setScreenFeedback: vi.fn(),
    refetchModels: vi.fn(async () => {}),
    ...overrides,
  });
}

function renderChatBar(overrides: Partial<ChatIntegration> = {}) {
  const { container } = render(
    <ChatBar mode="embedded" chatOrchestrator={createMockOrchestrator(overrides)} />
  );
  const textarea = container.querySelector("textarea") as HTMLTextAreaElement;
  const buttons = container.querySelectorAll("button[type='button']");
  const sendBtn = Array.from(buttons).find(b => b.textContent === "Send") as HTMLButtonElement;
  return { container, textarea, sendBtn };
}

describe("ChatBar with injected orchestrator", () => {
  it("renders textarea and send button", () => {
    const { textarea, sendBtn } = renderChatBar();
    expect(textarea).toBeInTheDocument();
    expect(sendBtn).toHaveTextContent("Send");
  });

  it("displays current value in textarea", () => {
    const { textarea } = renderChatBar({ value: "hello" });
    expect(textarea).toHaveValue("hello");
  });

  it("disables send button when isSending is true", () => {
    const { sendBtn } = renderChatBar({ isSending: true });
    expect(sendBtn).toBeDisabled();
  });

  it("calls submit on send button click", () => {
    const submit = vi.fn(async () => {});
    const { sendBtn } = renderChatBar({ submit });
    fireEvent.click(sendBtn);
    expect(submit).toHaveBeenCalledTimes(1);
  });

  it("calls submit on Enter key in textarea", () => {
    const submit = vi.fn(async () => {});
    const { textarea } = renderChatBar({ submit });
    fireEvent.keyDown(textarea, { key: "Enter" });
    expect(submit).toHaveBeenCalledTimes(1);
  });

  it("does not submit on Shift+Enter", () => {
    const submit = vi.fn(async () => {});
    const { textarea } = renderChatBar({ submit });
    fireEvent.keyDown(textarea, { key: "Enter", shiftKey: true });
    expect(submit).not.toHaveBeenCalled();
  });

  it("calls handleHistory on ArrowUp/Down", () => {
    const handleHistory = vi.fn();
    const { textarea } = renderChatBar({ handleHistory });
    fireEvent.keyDown(textarea, { key: "ArrowUp" });
    fireEvent.keyDown(textarea, { key: "ArrowDown" });
    expect(handleHistory).toHaveBeenCalledWith("up");
    expect(handleHistory).toHaveBeenCalledWith("down");
  });

  it("renders in embedded mode without fixed positioning", () => {
    const { container } = renderChatBar();
    const wrapper = container.firstElementChild as HTMLElement;
    expect(wrapper.className).not.toContain("fixed");
  });
});
