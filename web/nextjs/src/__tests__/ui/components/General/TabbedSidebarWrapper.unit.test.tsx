import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import TabbedSidebarWrapper from "@/ui/components/General/TabbedSidebarWrapper";

describe("TabbedSidebarWrapper", () => {
  it("renders the default tab and switches content when a different tab is selected", () => {
    const tabs = [
      { id: "one", label: "One", content: <div>First panel</div> },
      { id: "two", label: "Two", content: <div>Second panel</div> },
    ];

    render(<TabbedSidebarWrapper tabs={tabs} defaultTabId="two" className="test-class" />);

    expect(screen.getByRole("button", { name: "One" })).toHaveAttribute("aria-pressed", "false");
    expect(screen.getByRole("button", { name: "Two" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByText("Second panel")).toBeInTheDocument();
    expect(screen.getByRole("complementary")).toHaveClass("test-class");

    fireEvent.click(screen.getByRole("button", { name: "One" }));
    expect(screen.getByRole("button", { name: "One" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByText("First panel")).toBeInTheDocument();
  });

  it("falls back to the first tab when the default tab id is invalid", () => {
    render(
      <TabbedSidebarWrapper
        tabs={[
          { id: "alpha", label: "Alpha", content: <div>Alpha panel</div> },
          { id: "beta", label: "Beta", content: <div>Beta panel</div> },
        ]}
        defaultTabId="missing"
      />
    );

    expect(screen.getByRole("button", { name: "Alpha" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByText("Alpha panel")).toBeInTheDocument();
  });
});
