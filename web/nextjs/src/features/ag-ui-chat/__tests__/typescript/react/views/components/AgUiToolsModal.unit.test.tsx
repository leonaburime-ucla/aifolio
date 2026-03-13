import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import AgUiToolsModal from "@/features/ag-ui-chat/typescript/react/views/components/AgUiToolsModal";

describe("AgUiToolsModal", () => {
  it("hides switch_ag_ui_tab from the tools modal list", () => {
    render(<AgUiToolsModal activeTab="agentic-research" />);

    fireEvent.click(screen.getByRole("button", { name: "Show Tools" }));

    expect(screen.queryByText("switch_ag_ui_tab")).toBeNull();
    expect(screen.getByText("ar-set_active_dataset")).toBeInTheDocument();
  });
});
