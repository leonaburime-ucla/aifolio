import { describe, expect, it } from "vitest";
import {
  FALLBACK_CHAT_MODELS,
  resolveFallbackModelSelection,
} from "@/features/ai-chat/typescript/logic/modelSelection.logic";

describe("DR-005 fallback model order stability", () => {
  it("keeps fallback model order stable across calls", () => {
    const expectedOrder = FALLBACK_CHAT_MODELS.map((model) => model.id);

    const first = resolveFallbackModelSelection({ selectedModelId: null });
    const second = resolveFallbackModelSelection({ selectedModelId: null });

    expect(first.modelOptions.map((model) => model.id)).toEqual(expectedOrder);
    expect(second.modelOptions.map((model) => model.id)).toEqual(expectedOrder);
  });
});
