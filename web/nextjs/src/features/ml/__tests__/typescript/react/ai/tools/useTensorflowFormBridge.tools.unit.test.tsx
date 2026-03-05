import { renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useTensorflowFormBridge } from "@/features/ml/typescript/react/ai/tools/useTensorflowFormBridge.tools";
import type { TensorflowFormBridgeBindings } from "@/features/ml/__types__/typescript/react/ai/tools/tensorflowFormBridge.tools.types";

function createBindings(overrides: Partial<TensorflowFormBridgeBindings> = {}): TensorflowFormBridgeBindings {
  return {
    trainingMode: "wide_and_deep",
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
    onTrainClick: vi.fn(async () => undefined),
    ...overrides,
  };
}

describe("useTensorflowFormBridge", () => {
  afterEach(() => {
    delete window.__AIFOLIO_TENSORFLOW_FORM_BRIDGE__;
    vi.unstubAllGlobals();
  });

  it("registers and cleans up bridge on mount/unmount", () => {
    const bindings = createBindings();
    const { unmount } = renderHook(() => useTensorflowFormBridge(bindings));

    expect(window.__AIFOLIO_TENSORFLOW_FORM_BRIDGE__).toBeDefined();
    unmount();
    expect(window.__AIFOLIO_TENSORFLOW_FORM_BRIDGE__).toBeUndefined();
  });

  it("applies patch keys to bound setters", () => {
    const bindings = createBindings();
    renderHook(() => useTensorflowFormBridge(bindings));

    const result = window.__AIFOLIO_TENSORFLOW_FORM_BRIDGE__?.applyPatch({
      training_mode: "entity_embeddings",
      target_column: "target",
      task: "classification",
      epoch_values: [10, 20],
      batch_sizes: 64,
      learning_rates: [0.001, 0.0005],
      test_sizes: "0.2",
      hidden_dims: [128, 256],
      num_hidden_layers: [2, 3],
      dropouts: [0.1, 0.2],
      exclude_columns: ["id", "row_id"],
      date_columns: "Date",
      run_sweep: true,
      auto_distill: true,
      unknown_key: "x",
    } as never);

    expect(bindings.setTrainingMode).toHaveBeenCalledWith("entity_embeddings");
    expect(bindings.setTargetColumn).toHaveBeenCalledWith("target");
    expect(bindings.setTask).toHaveBeenCalledWith("classification");
    expect(bindings.setEpochValuesInput).toHaveBeenCalledWith("10,20");
    expect(bindings.setBatchSizesInput).toHaveBeenCalledWith("64");
    expect(bindings.setLearningRatesInput).toHaveBeenCalledWith("0.001,0.0005");
    expect(bindings.setTestSizesInput).toHaveBeenCalledWith("0.2");
    expect(bindings.setHiddenDimsInput).toHaveBeenCalledWith("128,256");
    expect(bindings.setNumHiddenLayersInput).toHaveBeenCalledWith("2,3");
    expect(bindings.setDropoutsInput).toHaveBeenCalledWith("0.1,0.2");
    expect(bindings.setExcludeColumnsInput).toHaveBeenCalledWith("id,row_id");
    expect(bindings.setDateColumnsInput).toHaveBeenCalledWith("Date");
    expect(bindings.toggleRunSweep).toHaveBeenCalledWith(true);
    expect(bindings.setAutoDistillEnabled).toHaveBeenCalledWith(true);
    expect(result?.applied).toContain("auto_distill");
    expect(result?.skipped).toContain("unknown_key");
  });

  it("starts training via startTrainingRuns after ui flush", async () => {
    vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) => {
      callback(0);
      return 1;
    });
    const onTrainClick = vi.fn(async () => undefined);
    const bindings = createBindings({ onTrainClick });
    renderHook(() => useTensorflowFormBridge(bindings));

    const result = await window.__AIFOLIO_TENSORFLOW_FORM_BRIDGE__?.startTrainingRuns();

    expect(result).toEqual({ status: "ok" });
    expect(onTrainClick).toHaveBeenCalledTimes(1);
  });
});
