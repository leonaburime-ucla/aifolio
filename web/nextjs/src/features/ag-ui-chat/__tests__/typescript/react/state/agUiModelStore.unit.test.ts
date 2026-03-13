import { describe, expect, it } from "vitest";
import { useAgUiModelStore } from "@/features/ag-ui-chat/typescript/react/state/zustand/agUiModelStore";

describe("useAgUiModelStore", () => {
  it("defaults the selected AG-UI model to Gemini 3 Flash Preview", () => {
    expect(useAgUiModelStore.getState().selectedModelId).toBe("gemini-3-flash-preview");
  });
});
