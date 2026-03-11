import { describe, expect, it } from "vitest";
import {
  getPytorchModeExplainer,
  getTensorflowModeExplainer,
} from "@/features/ml/typescript/config/trainingModeExplainers";

describe("trainingModeExplainers", () => {
  it("returns the registered PyTorch explainer for known modes", () => {
    expect(getPytorchModeExplainer("tabresnet").what).toContain("TabResNet");
  });

  it("falls back safely for unknown PyTorch modes", () => {
    expect(getPytorchModeExplainer("unsupported_mode").what).toContain("available");
  });

  it("returns the registered TensorFlow explainer for known modes", () => {
    expect(getTensorflowModeExplainer("entity_embeddings").what).toContain(
      "Entity embeddings"
    );
  });

  it("falls back safely for unknown TensorFlow modes", () => {
    expect(getTensorflowModeExplainer("unsupported_mode").why).toContain("generic");
  });
});
