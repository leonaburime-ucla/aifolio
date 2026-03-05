import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import CopilotChatSidebar from "@/features/ai-chat/typescript/react/views/components/CopilotChatSidebar";

const copilotChatMock = vi.fn();

vi.mock("@copilotkit/react-ui", () => ({
  CopilotChat: (props: Record<string, unknown>) => {
    copilotChatMock(props);
    return <div data-testid="copilot-chat-mock" />;
  },
}));

describe("CopilotChatSidebar component", () => {
  it("renders CopilotChat with expected class and labels", () => {
    render(<CopilotChatSidebar />);

    expect(screen.getByTestId("copilot-chat-mock")).toBeInTheDocument();
    expect(copilotChatMock).toHaveBeenCalledWith(
      expect.objectContaining({
        className: "h-full",
        labels: {
          title: "AI Chat",
          initial: "Ask a question to get started.",
        },
      })
    );
  });
});
