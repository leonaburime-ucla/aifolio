import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.mock("@copilotkit/react-core", () => ({
  CopilotKit: ({
    children,
    runtimeUrl,
    agent,
  }: {
    children: React.ReactNode;
    runtimeUrl: string;
    agent: string;
  }) => (
    <div
      data-testid="copilotkit"
      data-agent={agent}
      data-runtime-url={runtimeUrl}
    >
      {children}
    </div>
  ),
}));

import CopilotKitProvider from "@/ui/providers/CopilotKitProvider";

describe("CopilotKitProvider", () => {
  it("wraps children with the expected CopilotKit config", () => {
    render(
      <CopilotKitProvider>
        <div>child</div>
      </CopilotKitProvider>
    );

    expect(screen.getByTestId("copilotkit")).toHaveAttribute(
      "data-runtime-url",
      "/api/copilotkit"
    );
    expect(screen.getByTestId("copilotkit")).toHaveAttribute(
      "data-agent",
      "agentic-research"
    );
    expect(screen.getByText("child")).toBeInTheDocument();
  });
});
