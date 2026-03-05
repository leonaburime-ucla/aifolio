import { describe, expect, it } from "vitest";
import { getAgUiToolsForTab } from "@/features/ag-ui-chat/typescript/logic/agUiToolsCatalog.logic";

describe("getAgUiToolsForTab", () => {
  it("includes pytorch tools on pytorch tab", () => {
    const tools = getAgUiToolsForTab("pytorch");
    const names = tools.map((t) => t.name);
    expect(names).toContain("set_pytorch_form_fields");
    expect(names).toContain("start_pytorch_training_runs");
  });

  it("includes agentic research tools on agentic-research tab", () => {
    const tools = getAgUiToolsForTab("agentic-research");
    const names = tools.map((t) => t.name);
    expect(names).toContain("ar-set_active_dataset");
    expect(names).toContain("ar-reorder_chart_specs");
  });

  it("includes tensorflow tools on tensorflow tab", () => {
    const tools = getAgUiToolsForTab("tensorflow");
    const names = tools.map((t) => t.name);
    expect(names).toContain("set_tensorflow_form_fields");
    expect(names).toContain("start_tensorflow_training_runs");
    expect(names).not.toContain("add_chart_spec");
    expect(names).not.toContain("clear_charts");
  });
});
