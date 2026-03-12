import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import ChatSidebar from "@/features/ai-chat/typescript/react/views/components/ChatSidebar";
import type { ChatOrchestrator } from "@/features/ai-chat/typescript/react/orchestrators/chatOrchestrator";

const sidebarUiMock = vi.fn();

vi.mock("@/features/ai-chat/typescript/react/hooks/useChatSidebar.web", () => ({
  useChatSidebarUi: (...args: unknown[]) => sidebarUiMock(...args),
}));

vi.mock("@/features/ai-chat/typescript/react/views/components/ChatBar", () => ({
  default: () => <div data-testid="chat-bar-mock">ChatBar</div>,
}));

function createOrchestrator(overrides: Partial<ChatOrchestrator> = {}): ChatOrchestrator {
  return {
    messages: [],
    inputHistory: [],
    historyCursor: null,
    isSending: false,
    modelOptions: [],
    selectedModelId: null,
    isModelsLoading: false,
    screenFeedback: null,
    activeDatasetId: null,
    value: "",
    showTooltip: false,
    attachments: [],
    setShowTooltip: vi.fn(),
    setValue: vi.fn(),
    resetValue: vi.fn(),
    addAttachments: vi.fn(),
    clearAttachments: vi.fn(),
    removeAttachment: vi.fn(),
    submit: vi.fn(async () => undefined),
    retryLastSubmission: vi.fn(async () => undefined),
    handleHistory: vi.fn(),
    resetHistoryCursor: vi.fn(),
    setSelectedModelId: vi.fn(),
    refetchModels: vi.fn(async () => undefined),
    setScreenFeedback: vi.fn(),
    ...overrides,
  };
}

describe("ChatSidebar component", () => {
  afterEach(() => {
    cleanup();
  });

  it("renders empty state and loading-model placeholder", () => {
    const orchestrator = createOrchestrator({
      modelOptions: [],
      isModelsLoading: true,
    });
    sidebarUiMock.mockReturnValue({
      scrollRef: { current: null },
      isDragging: false,
      copiedId: null,
      handleCopy: vi.fn(),
      handleDrop: vi.fn(),
      handleDragOver: vi.fn(),
      handleDragLeave: vi.fn(),
    });

    render(<ChatSidebar chatOrchestrator={() => orchestrator} />);

    expect(screen.getByText("Ask a question to get started.")).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "Loading models..." })).toBeInTheDocument();
    expect(screen.getByTestId("chat-bar-mock")).toBeInTheDocument();
  });

  it("renders messages, supports copy action, and updates selected model", () => {
    const handleCopy = vi.fn(async () => undefined);
    const setSelectedModelId = vi.fn();
    const orchestrator = createOrchestrator({
      messages: [
        { id: "u1", role: "user", content: "hello", createdAt: 1 },
        { id: "a1", role: "assistant", content: "**hi**", createdAt: 2 },
      ],
      modelOptions: [
        { id: "gemini-a", label: "Gemini A" },
        { id: "gemini-b", label: "Gemini B" },
      ],
      selectedModelId: "gemini-a",
      setSelectedModelId,
      attachments: [{ name: "file.txt", type: "text/plain", size: 10, dataUrl: "data:" }],
      removeAttachment: vi.fn(),
    });
    sidebarUiMock.mockReturnValue({
      scrollRef: { current: null },
      isDragging: false,
      copiedId: null,
      handleCopy,
      handleDrop: vi.fn(),
      handleDragOver: vi.fn(),
      handleDragLeave: vi.fn(),
    });

    render(<ChatSidebar chatOrchestrator={() => orchestrator} />);

    expect(screen.getByText("hello")).toBeInTheDocument();
    expect(screen.getByText("hi")).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Select AI model"), {
      target: { value: "gemini-b" },
    });
    expect(setSelectedModelId).toHaveBeenCalledWith("gemini-b");

    fireEvent.change(screen.getByLabelText("Select AI model"), {
      target: { value: "" },
    });
    expect(setSelectedModelId).toHaveBeenCalledWith(null);

    fireEvent.click(screen.getAllByRole("button", { name: "Copy" })[0]);
    expect(handleCopy).toHaveBeenCalledWith("u1", "hello");

    fireEvent.click(screen.getByRole("button", { name: "Remove attachment" }));
    expect(orchestrator.removeAttachment).toHaveBeenCalledWith(0);
  });

  it("shows drag overlay when sidebar UI reports dragging", () => {
    const orchestrator = createOrchestrator();
    sidebarUiMock.mockReturnValue({
      scrollRef: { current: null },
      isDragging: true,
      copiedId: null,
      handleCopy: vi.fn(),
      handleDrop: vi.fn(),
      handleDragOver: vi.fn(),
      handleDragLeave: vi.fn(),
    });

    render(<ChatSidebar chatOrchestrator={() => orchestrator} />);
    expect(screen.getByText("Drop files to attach")).toBeInTheDocument();
  });

  it("renders working indicator while sending", () => {
    const orchestrator = createOrchestrator({ isSending: true });
    sidebarUiMock.mockReturnValue({
      scrollRef: { current: null },
      isDragging: false,
      copiedId: null,
      handleCopy: vi.fn(),
      handleDrop: vi.fn(),
      handleDragOver: vi.fn(),
      handleDragLeave: vi.fn(),
    });

    render(<ChatSidebar chatOrchestrator={() => orchestrator} />);
    expect(screen.getByText("Working")).toBeInTheDocument();
  });

  it("renders copied checkmark when copiedId matches message", () => {
    const orchestrator = createOrchestrator({
      messages: [{ id: "u1", role: "user", content: "hello", createdAt: 1 }],
    });
    sidebarUiMock.mockReturnValue({
      scrollRef: { current: null },
      isDragging: false,
      copiedId: "u1",
      handleCopy: vi.fn(),
      handleDrop: vi.fn(),
      handleDragOver: vi.fn(),
      handleDragLeave: vi.fn(),
    });

    render(<ChatSidebar chatOrchestrator={() => orchestrator} />);
    expect(screen.getByRole("button", { name: "✓" })).toBeInTheDocument();
  });

  it("renders persistent feedback and dismisses it through the orchestrator", () => {
    const setScreenFeedback = vi.fn();
    const retryLastSubmission = vi.fn(async () => undefined);
    const orchestrator = createOrchestrator({
      screenFeedback: {
        kind: "error",
        code: "CHAT_REQUEST_FAILED",
        message: "Could not reach the AI service.",
        retryable: true,
        actionLabel: "Try again",
      },
      setScreenFeedback,
      retryLastSubmission,
    });
    sidebarUiMock.mockReturnValue({
      scrollRef: { current: null },
      isDragging: false,
      copiedId: null,
      handleCopy: vi.fn(),
      handleDrop: vi.fn(),
      handleDragOver: vi.fn(),
      handleDragLeave: vi.fn(),
    });

    render(<ChatSidebar chatOrchestrator={() => orchestrator} />);

    expect(screen.getByText("Could not reach the AI service.")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(retryLastSubmission).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByRole("button", { name: "Dismiss feedback" }));
    expect(setScreenFeedback).toHaveBeenCalledWith(null);
  });
});
