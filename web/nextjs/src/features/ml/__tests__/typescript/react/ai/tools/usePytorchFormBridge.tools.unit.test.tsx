import { renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { usePytorchFormBridge } from "@/features/ml/typescript/react/ai/tools/usePytorchFormBridge.tools";
import type { PytorchFormBridgeBindings } from "@/features/ml/__types__/typescript/react/ai/tools/pytorchFormBridge.tools.types";

function createBindings(overrides: Partial<PytorchFormBridgeBindings> = {}): PytorchFormBridgeBindings {
  return {
    trainingMode: "mlp_dense",
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
    onTrainClick: vi.fn(async () => undefined),
    ...overrides,
  };
}

describe("usePytorchFormBridge", () => {
  afterEach(() => {
    delete window.__AIFOLIO_PYTORCH_FORM_BRIDGE__;
    vi.unstubAllGlobals();
  });

  it("registers and cleans up bridge on mount/unmount", () => {
    const bindings = createBindings();
    const { unmount } = renderHook(() => usePytorchFormBridge(bindings));

    expect(window.__AIFOLIO_PYTORCH_FORM_BRIDGE__).toBeDefined();

    unmount();
    expect(window.__AIFOLIO_PYTORCH_FORM_BRIDGE__).toBeUndefined();
  });

  it("applies patch keys to bound setters", () => {
    const bindings = createBindings();
    renderHook(() => usePytorchFormBridge(bindings));

    const result = window.__AIFOLIO_PYTORCH_FORM_BRIDGE__?.applyPatch({
      dataset_id: "fraud_detection_phishing_websites.csv",
      training_mode: "tabresnet",
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

    expect(bindings.setDatasetId).toHaveBeenCalledWith("fraud_detection_phishing_websites.csv");
    expect(bindings.setTrainingMode).toHaveBeenCalledWith("tabresnet");
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
    expect(result?.applied).toContain("dataset_id");
    expect(result?.applied).toContain("auto_distill");
    expect(result?.skipped).toContain("unknown_key");
  });

  it("does not toggle sweep/distill when requested values already match", () => {
    const bindings = createBindings({
      runSweepEnabled: true,
      autoDistillEnabled: true,
    });
    renderHook(() => usePytorchFormBridge(bindings));

    window.__AIFOLIO_PYTORCH_FORM_BRIDGE__?.applyPatch({
      run_sweep: true,
      auto_distill: true,
    });

    expect(bindings.toggleRunSweep).not.toHaveBeenCalled();
    expect(bindings.setAutoDistillEnabled).not.toHaveBeenCalled();
  });

  it("starts training via startTrainingRuns after ui flush", async () => {
    vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) => {
      callback(0);
      return 1;
    });
    const onTrainClick = vi.fn(async () => undefined);
    const bindings = createBindings({ onTrainClick });
    renderHook(() => usePytorchFormBridge(bindings));

    const result = await window.__AIFOLIO_PYTORCH_FORM_BRIDGE__?.startTrainingRuns();

    expect(result).toEqual({ status: "ok" });
    expect(onTrainClick).toHaveBeenCalledTimes(1);
  });

  it("uses latest onTrainClick after rerender", async () => {
    vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) => {
      callback(0);
      return 1;
    });
    const firstTrain = vi.fn(async () => undefined);
    const secondTrain = vi.fn(async () => undefined);

    const { rerender } = renderHook(
      ({ bindings }) => usePytorchFormBridge(bindings),
      { initialProps: { bindings: createBindings({ onTrainClick: firstTrain }) } },
    );

    rerender({ bindings: createBindings({ onTrainClick: secondTrain }) });
    const result = await window.__AIFOLIO_PYTORCH_FORM_BRIDGE__?.startTrainingRuns();

    expect(result).toEqual({ status: "ok" });
    expect(firstTrain).not.toHaveBeenCalled();
    expect(secondTrain).toHaveBeenCalledTimes(1);
  });
});
