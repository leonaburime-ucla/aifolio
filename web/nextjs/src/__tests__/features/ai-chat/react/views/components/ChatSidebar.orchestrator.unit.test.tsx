import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import ChatSidebar from "@/features/ai-chat/react/views/components/ChatSidebar";
import type { ChatIntegration } from "@aifolio/contracts/entities/chat";
import type { ChatSidebarUi } from "@/features/ai-chat/react/hooks/useChatSidebar.web";

afterEach(cleanup);

vi.mock("@/features/ai-chat/react/views/components/ChatBar", () => ({
  default: () => <div data-testid="chat-bar-mock">ChatBar</div>,
}));

function createMockSidebarUi(): () => ChatSidebarUi {
  return () => ({
    scrollRef: { current: null },
    isDragging: false,
    copiedId: null,
    handleCopy: vi.fn(async () => {}),
    handleDrop: vi.fn(async () => {}),
    handleDragOver: vi.fn(),
    handleDragLeave: vi.fn(),
  });
}

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

describe("ChatSidebar with injected orchestrator", () => {
  const useSidebarUi = createMockSidebarUi();

  it("renders header and model selector", () => {
    render(<ChatSidebar chatOrchestrator={createMockOrchestrator()} useSidebarUi={useSidebarUi} />);
    expect(screen.getByText("AI Chat")).toBeInTheDocument();
    expect(screen.getByLabelText("Select AI model")).toBeInTheDocument();
  });

  it("shows empty state when no messages", () => {
    render(<ChatSidebar chatOrchestrator={createMockOrchestrator()} useSidebarUi={useSidebarUi} />);
    expect(screen.getByText("Ask a question to get started.")).toBeInTheDocument();
  });

  it("renders user and assistant messages", () => {
    const messages = [
      { id: "1", role: "user" as const, content: "Hello" },
      { id: "2", role: "assistant" as const, content: "Hi there" },
    ];
    render(<ChatSidebar chatOrchestrator={createMockOrchestrator({ messages })} useSidebarUi={useSidebarUi} />);
    expect(screen.getByText("Hello")).toBeInTheDocument();
    expect(screen.getByText("Hi there")).toBeInTheDocument();
  });

  it("shows loading spinner when sending", () => {
    render(<ChatSidebar chatOrchestrator={createMockOrchestrator({ isSending: true })} useSidebarUi={useSidebarUi} />);
    expect(screen.getByText("Working")).toBeInTheDocument();
  });

  it("renders model options in selector", () => {
    const modelOptions = [
      { id: "gemini-flash", label: "Gemini Flash" },
      { id: "gemini-pro", label: "Gemini Pro" },
    ];
    render(
      <ChatSidebar
        chatOrchestrator={createMockOrchestrator({
          modelOptions,
          selectedModelId: "gemini-flash",
        })}
        useSidebarUi={useSidebarUi}
      />
    );
    const options = screen.getAllByRole("option");
    expect(options).toHaveLength(2);
    expect(options[0]).toHaveTextContent("Gemini Flash");
  });

  it("disables model selector when loading", () => {
    render(
      <ChatSidebar chatOrchestrator={createMockOrchestrator({ isModelsLoading: true })} useSidebarUi={useSidebarUi} />
    );
    expect(screen.getByLabelText("Select AI model")).toBeDisabled();
  });

  it("renders attachments section when attachments exist", () => {
    const attachments = [{ name: "file.txt", size: 100, type: "text/plain" }];
    render(
      <ChatSidebar chatOrchestrator={createMockOrchestrator({ attachments } as any)} useSidebarUi={useSidebarUi} />
    );
    expect(screen.getByText("file.txt")).toBeInTheDocument();
  });
});
