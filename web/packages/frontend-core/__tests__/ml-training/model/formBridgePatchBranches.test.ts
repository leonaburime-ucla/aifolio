import { describe, expect, it, vi } from "vitest";
import { applyMlFormBridgePatch, toBridgeCsv } from "../../../src/ml-training/model/formBridgePatch";

function createBindings() {
  return {
    setDatasetId: vi.fn(),
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
  };
}

describe("toBridgeCsv edge cases", () => {
  it("handles undefined", () => {
    expect(toBridgeCsv(undefined)).toBe("");
  });

  it("handles single number", () => {
    expect(toBridgeCsv(0.001)).toBe("0.001");
  });

  it("filters empty strings from arrays", () => {
    expect(toBridgeCsv(["", "  ", "valid"])).toBe("valid");
  });
});

describe("applyMlFormBridgePatch — all numeric field branches", () => {
  it("applies batch_sizes", () => {
    const bindings = createBindings();
    const result = applyMlFormBridgePatch({ batch_sizes: [32, 64] } as never, bindings);
    expect(bindings.setBatchSizesInput).toHaveBeenCalledWith("32,64");
    expect(result.applied).toContain("batch_sizes");
  });

  it("applies learning_rates", () => {
    const bindings = createBindings();
    const result = applyMlFormBridgePatch({ learning_rates: [0.001, 0.01] } as never, bindings);
    expect(bindings.setLearningRatesInput).toHaveBeenCalledWith("0.001,0.01");
    expect(result.applied).toContain("learning_rates");
  });

  it("applies test_sizes", () => {
    const bindings = createBindings();
    const result = applyMlFormBridgePatch({ test_sizes: 0.2 } as never, bindings);
    expect(bindings.setTestSizesInput).toHaveBeenCalledWith("0.2");
    expect(result.applied).toContain("test_sizes");
  });

  it("applies hidden_dims", () => {
    const bindings = createBindings();
    const result = applyMlFormBridgePatch({ hidden_dims: [128, 256] } as never, bindings);
    expect(bindings.setHiddenDimsInput).toHaveBeenCalledWith("128,256");
    expect(result.applied).toContain("hidden_dims");
  });

  it("applies num_hidden_layers", () => {
    const bindings = createBindings();
    const result = applyMlFormBridgePatch({ num_hidden_layers: 3 } as never, bindings);
    expect(bindings.setNumHiddenLayersInput).toHaveBeenCalledWith("3");
    expect(result.applied).toContain("num_hidden_layers");
  });

  it("applies dropouts", () => {
    const bindings = createBindings();
    const result = applyMlFormBridgePatch({ dropouts: [0.1, 0.2] } as never, bindings);
    expect(bindings.setDropoutsInput).toHaveBeenCalledWith("0.1,0.2");
    expect(result.applied).toContain("dropouts");
  });

  it("applies exclude_columns", () => {
    const bindings = createBindings();
    const result = applyMlFormBridgePatch({ exclude_columns: ["id", "name"] } as never, bindings);
    expect(bindings.setExcludeColumnsInput).toHaveBeenCalledWith("id,name");
    expect(result.applied).toContain("exclude_columns");
  });

  it("applies date_columns", () => {
    const bindings = createBindings();
    const result = applyMlFormBridgePatch({ date_columns: ["created_at"] } as never, bindings);
    expect(bindings.setDateColumnsInput).toHaveBeenCalledWith("created_at");
    expect(result.applied).toContain("date_columns");
  });
});

describe("applyMlFormBridgePatch — set_sweep_values fallback", () => {
  it("uses set_sweep_values when run_sweep is undefined", () => {
    const bindings = createBindings();
    const result = applyMlFormBridgePatch({ set_sweep_values: true } as never, bindings);
    expect(bindings.toggleRunSweep).toHaveBeenCalledWith(true);
    expect(result.applied).toContain("run_sweep");
  });

  it("prefers run_sweep over set_sweep_values when both present", () => {
    const bindings = createBindings();
    applyMlFormBridgePatch({ run_sweep: false, set_sweep_values: true } as never, bindings);
    expect(bindings.toggleRunSweep).not.toHaveBeenCalled();
  });
});

describe("applyMlFormBridgePatch — auto_distill toggling", () => {
  it("toggles auto_distill on when currently off", () => {
    const bindings = createBindings();
    const result = applyMlFormBridgePatch({ auto_distill: true } as never, bindings);
    expect(bindings.setAutoDistillEnabled).toHaveBeenCalledWith(true);
    expect(result.applied).toContain("auto_distill");
  });

  it("toggles auto_distill off when currently on", () => {
    const bindings = createBindings();
    bindings.autoDistillEnabled = true;
    const result = applyMlFormBridgePatch({ auto_distill: false } as never, bindings);
    expect(bindings.setAutoDistillEnabled).toHaveBeenCalledWith(false);
    expect(result.applied).toContain("auto_distill");
  });
});

describe("applyMlFormBridgePatch — empty patch", () => {
  it("returns empty applied and skipped for empty patch", () => {
    const bindings = createBindings();
    const result = applyMlFormBridgePatch({} as never, bindings);
    expect(result.applied).toEqual([]);
    expect(result.skipped).toEqual([]);
  });
});
