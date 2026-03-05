import { describe, expect, it } from "vitest";
import { toDatasetOptions } from "@/features/agentic-research/typescript/logic/agenticResearchManifest.logic";

describe("DR-001 dataset options mapping order", () => {
  it("maps id/label/description in manifest order only", () => {
    const options = toDatasetOptions({
      datasetManifest: [
        { id: "b", label: "B", description: "second" },
        { id: "a", label: "A" },
      ],
    });

    expect(options).toEqual([
      { id: "b", label: "B", description: "second" },
      { id: "a", label: "A", description: undefined },
    ]);
  });
});
