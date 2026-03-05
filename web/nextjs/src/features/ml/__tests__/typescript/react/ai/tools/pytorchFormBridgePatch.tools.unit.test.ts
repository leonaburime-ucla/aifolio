import { describe, expect, it, vi } from "vitest";
import { applyPytorchBridgePatch, toBridgeCsv } from "@/features/ml/typescript/react/ai/tools/pytorchFormBridgePatch.tools";
import type { PytorchBridgePatchBindings } from "@/features/ml/__types__/typescript/react/ai/tools/pytorchFormBridge.tools.types";

function createBindings(overrides: Partial<PytorchBridgePatchBindings> = {}): PytorchBridgePatchBindings {
  return {
    setTrainingMode: vi.fn(),
    setTargetColumn: vi.fn(),
    setTask: vi.fn(),
    runSweepEnabled: false,
    toggleRunSweep: vi.fn(),
    setEpochValuesInput: vi.fn(),
    setBatchSizesInput: vi.fn(),
    setLearningRatesInput: vi.fn(),
    setTestSizesInput: vi.fn(),
    setHiddenDimsInput: vi.fn(),
    setNumHiddenLayersInput: vi.fn(),
    setDropoutsInput: vi.fn(),
    setExcludeColumnsInput: vi.fn(),
    setDateColumnsInput: vi.fn(),
    autoDistillEnabled: false,
    setAutoDistillEnabled: vi.fn(),
    ...overrides,
  };
}

describe("pytorchFormBridgePatch.tools", () => {
  it("normalizes csv bridge values", () => {
    expect(toBridgeCsv(["a", " b ", "", 1])).toBe("a,b,1");
    expect(toBridgeCsv(64)).toBe("64");
    expect(toBridgeCsv(null)).toBe("");
  });

  it("applies patch deterministically and reports skipped keys", () => {
    const bindings = createBindings();
    const result = applyPytorchBridgePatch(
      {
        training_mode: "mlp_dense",
        epoch_values: [10, 20],
        run_sweep: true,
        auto_distill: true,
        unknown_key: "x",
      } as never,
      bindings
    );

    expect(bindings.setTrainingMode).toHaveBeenCalledWith("mlp_dense");
    expect(bindings.setEpochValuesInput).toHaveBeenCalledWith("10,20");
    expect(bindings.toggleRunSweep).toHaveBeenCalledWith(true);
    expect(bindings.setAutoDistillEnabled).toHaveBeenCalledWith(true);
    expect(result.applied).toEqual(
      expect.arrayContaining(["training_mode", "epoch_values", "run_sweep", "auto_distill"])
    );
    expect(result.skipped).toContain("unknown_key");
  });

  it("does not toggle sweep/distill when values already match", () => {
    const bindings = createBindings({
      runSweepEnabled: true,
      autoDistillEnabled: true,
    });

    applyPytorchBridgePatch(
      {
        run_sweep: true,
        auto_distill: true,
      },
      bindings
    );

    expect(bindings.toggleRunSweep).not.toHaveBeenCalled();
    expect(bindings.setAutoDistillEnabled).not.toHaveBeenCalled();
  });
});

