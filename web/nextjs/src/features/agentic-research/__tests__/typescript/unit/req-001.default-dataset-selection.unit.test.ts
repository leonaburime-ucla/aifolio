import { describe, expect, it } from "vitest";
import { resolveDefaultDatasetId } from "@/features/agentic-research/typescript/logic/agenticResearchManifest.logic";

describe("REQ-001 default dataset selection", () => {
  it("uses selectedDatasetId, then first dataset id, then null", () => {
    expect(
      resolveDefaultDatasetId({
        selectedDatasetId: "already-selected",
        datasets: [{ id: "a", label: "A" }],
      })
    ).toBe("already-selected");

    expect(
      resolveDefaultDatasetId({
        selectedDatasetId: null,
        datasets: [{ id: "a", label: "A" }, { id: "b", label: "B" }],
      })
    ).toBe("a");

    expect(
      resolveDefaultDatasetId({
        selectedDatasetId: null,
        datasets: [],
      })
    ).toBeNull();
  });
});
