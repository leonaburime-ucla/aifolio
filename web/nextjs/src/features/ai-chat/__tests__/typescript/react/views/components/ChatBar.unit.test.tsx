import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import ChatBar from "@/features/ai-chat/typescript/react/views/components/ChatBar";
import type { ChatOrchestrator } from "@/features/ai-chat/typescript/react/orchestrators/chatOrchestrator";

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

describe("ChatBar component", () => {
  afterEach(() => {
    vi.useRealTimers();
    cleanup();
  });

  it("disables Send while sending and submits when enabled", () => {
    const orchestrator = createOrchestrator({ isSending: true });
    render(<ChatBar chatOrchestrator={() => orchestrator} />);

    const sendButton = screen.getByRole("button", { name: "Send" });
    expect(sendButton).toBeDisabled();

    const enabledOrchestrator = createOrchestrator({ isSending: false });
    cleanup();
    render(<ChatBar chatOrchestrator={() => enabledOrchestrator} />);
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    expect(enabledOrchestrator.submit).toHaveBeenCalledTimes(1);
  });

  it("handles Enter submit and history arrow keys", () => {
    const orchestrator = createOrchestrator({ value: "hello" });
    render(<ChatBar chatOrchestrator={() => orchestrator} />);

    const input = screen.getByLabelText("Chat input");

    fireEvent.keyDown(input, { key: "Enter" });
    fireEvent.keyDown(input, { key: "ArrowUp" });
    fireEvent.keyDown(input, { key: "ArrowDown" });

    expect(orchestrator.submit).toHaveBeenCalledTimes(1);
    expect(orchestrator.handleHistory).toHaveBeenNthCalledWith(1, "up");
    expect(orchestrator.handleHistory).toHaveBeenNthCalledWith(2, "down");
  });

  it("updates value and resets history cursor on change", () => {
    const orchestrator = createOrchestrator({ value: "before" });
    render(<ChatBar chatOrchestrator={() => orchestrator} />);

    const input = screen.getByLabelText("Chat input");
    fireEvent.change(input, { target: { value: "after" } });

    expect(orchestrator.setValue).toHaveBeenCalledWith("after");
    expect(orchestrator.resetHistoryCursor).toHaveBeenCalledTimes(1);
  });

  it("shows tooltip interaction and renders embedded mode layout", () => {
    vi.useFakeTimers();
    const orchestrator = createOrchestrator();
    render(<ChatBar mode="embedded" chatOrchestrator={() => orchestrator} />);

    const plusButton = screen.getByRole("button", { name: "+" });
    fireEvent.mouseEnter(plusButton);
    expect(orchestrator.setShowTooltip).toHaveBeenCalledWith(true);

    fireEvent.click(plusButton);
    expect(orchestrator.setShowTooltip).toHaveBeenCalledWith(true);
    vi.advanceTimersByTime(1500);
    expect(orchestrator.setShowTooltip).toHaveBeenCalledWith(false);

    fireEvent.mouseLeave(plusButton);
    expect(orchestrator.setShowTooltip).toHaveBeenCalledWith(false);

    const input = screen.getByLabelText("Chat input");
    expect(input).toHaveAttribute("rows", "3");
  });

  it("renders visible tooltip class when orchestrator state sets showTooltip=true", () => {
    const orchestrator = createOrchestrator({ showTooltip: true });
    render(<ChatBar chatOrchestrator={() => orchestrator} />);

    const tooltip = screen.getByText("Disabled for now");
    expect(tooltip.className).toContain("opacity-100");
  });
});
