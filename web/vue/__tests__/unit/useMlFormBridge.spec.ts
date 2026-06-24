import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { ref } from "vue";
import { usePytorchFormBridge, useTensorflowFormBridge } from "~/features/ml/composables/useMlFormBridge";

let mountedCallbacks: (() => void)[] = [];
let unmountedCallbacks: (() => void)[] = [];

vi.mock("vue", async () => {
  const actual = await vi.importActual<typeof import("vue")>("vue");
  return {
    ...actual,
    onMounted: (cb: () => void) => { mountedCallbacks.push(cb); },
    onUnmounted: (cb: () => void) => { unmountedCallbacks.push(cb); },
  };
});

function createBindings() {
  return {
    selectedDatasetId: ref<string | null>("churn.csv"),
    trainingMode: ref("mlp_dense"),
    targetColumn: ref("target"),
    task: ref("auto"),
    sweepEnabled: ref(false),
    epochValues: ref("60"),
    batchSizes: ref("64"),
    learningRates: ref("0.001"),
    testSizes: ref("0.2"),
    hiddenDims: ref("128"),
    numHiddenLayers: ref("2"),
    dropouts: ref("0.1"),
    excludeColumns: ref("customerID"),
    dateColumns: ref(""),
    autoDistill: ref(false),
    onTrain: vi.fn().mockResolvedValue(undefined),
    onDatasetChange: vi.fn(),
  };
}

describe("usePytorchFormBridge", () => {
  beforeEach(() => {
    mountedCallbacks = [];
    unmountedCallbacks = [];
    delete (window as any).__AIFOLIO_PYTORCH_FORM_BRIDGE__;
  });

  afterEach(() => {
    delete (window as any).__AIFOLIO_PYTORCH_FORM_BRIDGE__;
  });

  it("registers bridge on mount", () => {
    const bindings = createBindings();
    usePytorchFormBridge(bindings);

    expect(window.__AIFOLIO_PYTORCH_FORM_BRIDGE__).toBeUndefined();
    mountedCallbacks.forEach((cb) => cb());
    expect(window.__AIFOLIO_PYTORCH_FORM_BRIDGE__).toBeDefined();
  });

  it("removes bridge on unmount", () => {
    const bindings = createBindings();
    usePytorchFormBridge(bindings);

    mountedCallbacks.forEach((cb) => cb());
    expect(window.__AIFOLIO_PYTORCH_FORM_BRIDGE__).toBeDefined();

    unmountedCallbacks.forEach((cb) => cb());
    expect(window.__AIFOLIO_PYTORCH_FORM_BRIDGE__).toBeUndefined();
  });

  describe("applyPatch", () => {
    it("applies training_mode patch", () => {
      const bindings = createBindings();
      usePytorchFormBridge(bindings);
      mountedCallbacks.forEach((cb) => cb());

      const bridge = window.__AIFOLIO_PYTORCH_FORM_BRIDGE__!;
      const result = bridge.applyPatch({ training_mode: "tabresnet" });

      expect(bindings.trainingMode.value).toBe("tabresnet");
      expect(result.applied).toContain("training_mode");
      expect(result.skipped).toHaveLength(0);
    });

    it("applies dataset_id patch via onDatasetChange", () => {
      const bindings = createBindings();
      usePytorchFormBridge(bindings);
      mountedCallbacks.forEach((cb) => cb());

      const bridge = window.__AIFOLIO_PYTORCH_FORM_BRIDGE__!;
      const result = bridge.applyPatch({ dataset_id: "fraud.csv" });

      expect(bindings.onDatasetChange).toHaveBeenCalledWith("fraud.csv");
      expect(result.applied).toContain("dataset_id");
    });

    it("applies epoch_values with array normalization", () => {
      const bindings = createBindings();
      usePytorchFormBridge(bindings);
      mountedCallbacks.forEach((cb) => cb());

      const bridge = window.__AIFOLIO_PYTORCH_FORM_BRIDGE__!;
      bridge.applyPatch({ epoch_values: [20, 40, 60] });

      expect(bindings.epochValues.value).toBe("20,40,60");
    });

    it("applies batch_sizes with scalar", () => {
      const bindings = createBindings();
      usePytorchFormBridge(bindings);
      mountedCallbacks.forEach((cb) => cb());

      const bridge = window.__AIFOLIO_PYTORCH_FORM_BRIDGE__!;
      bridge.applyPatch({ batch_sizes: "32,128" });

      expect(bindings.batchSizes.value).toBe("32,128");
    });

    it("applies target_column", () => {
      const bindings = createBindings();
      usePytorchFormBridge(bindings);
      mountedCallbacks.forEach((cb) => cb());

      const bridge = window.__AIFOLIO_PYTORCH_FORM_BRIDGE__!;
      bridge.applyPatch({ target_column: "price" });

      expect(bindings.targetColumn.value).toBe("price");
    });

    it("applies task", () => {
      const bindings = createBindings();
      usePytorchFormBridge(bindings);
      mountedCallbacks.forEach((cb) => cb());

      const bridge = window.__AIFOLIO_PYTORCH_FORM_BRIDGE__!;
      bridge.applyPatch({ task: "classification" });

      expect(bindings.task.value).toBe("classification");
    });

    it("applies run_sweep toggle", () => {
      const bindings = createBindings();
      usePytorchFormBridge(bindings);
      mountedCallbacks.forEach((cb) => cb());

      const bridge = window.__AIFOLIO_PYTORCH_FORM_BRIDGE__!;
      bridge.applyPatch({ run_sweep: true });

      expect(bindings.sweepEnabled.value).toBe(true);
    });

    it("applies set_sweep_values as alias for run_sweep", () => {
      const bindings = createBindings();
      usePytorchFormBridge(bindings);
      mountedCallbacks.forEach((cb) => cb());

      const bridge = window.__AIFOLIO_PYTORCH_FORM_BRIDGE__!;
      bridge.applyPatch({ set_sweep_values: true });

      expect(bindings.sweepEnabled.value).toBe(true);
    });

    it("applies learning_rates", () => {
      const bindings = createBindings();
      usePytorchFormBridge(bindings);
      mountedCallbacks.forEach((cb) => cb());

      const bridge = window.__AIFOLIO_PYTORCH_FORM_BRIDGE__!;
      bridge.applyPatch({ learning_rates: [0.0005, 0.001] });

      expect(bindings.learningRates.value).toBe("0.0005,0.001");
    });

    it("applies test_sizes", () => {
      const bindings = createBindings();
      usePytorchFormBridge(bindings);
      mountedCallbacks.forEach((cb) => cb());

      const bridge = window.__AIFOLIO_PYTORCH_FORM_BRIDGE__!;
      bridge.applyPatch({ test_sizes: [0.2, 0.3] });

      expect(bindings.testSizes.value).toBe("0.2,0.3");
    });

    it("applies hidden_dims", () => {
      const bindings = createBindings();
      usePytorchFormBridge(bindings);
      mountedCallbacks.forEach((cb) => cb());

      const bridge = window.__AIFOLIO_PYTORCH_FORM_BRIDGE__!;
      bridge.applyPatch({ hidden_dims: [64, 96] });

      expect(bindings.hiddenDims.value).toBe("64,96");
    });

    it("applies num_hidden_layers", () => {
      const bindings = createBindings();
      usePytorchFormBridge(bindings);
      mountedCallbacks.forEach((cb) => cb());

      const bridge = window.__AIFOLIO_PYTORCH_FORM_BRIDGE__!;
      bridge.applyPatch({ num_hidden_layers: "3" });

      expect(bindings.numHiddenLayers.value).toBe("3");
    });

    it("applies dropouts", () => {
      const bindings = createBindings();
      usePytorchFormBridge(bindings);
      mountedCallbacks.forEach((cb) => cb());

      const bridge = window.__AIFOLIO_PYTORCH_FORM_BRIDGE__!;
      bridge.applyPatch({ dropouts: [0.1, 0.2] });

      expect(bindings.dropouts.value).toBe("0.1,0.2");
    });

    it("applies exclude_columns", () => {
      const bindings = createBindings();
      usePytorchFormBridge(bindings);
      mountedCallbacks.forEach((cb) => cb());

      const bridge = window.__AIFOLIO_PYTORCH_FORM_BRIDGE__!;
      bridge.applyPatch({ exclude_columns: "id,timestamp" });

      expect(bindings.excludeColumns.value).toBe("id,timestamp");
    });

    it("applies date_columns", () => {
      const bindings = createBindings();
      usePytorchFormBridge(bindings);
      mountedCallbacks.forEach((cb) => cb());

      const bridge = window.__AIFOLIO_PYTORCH_FORM_BRIDGE__!;
      bridge.applyPatch({ date_columns: "created_at" });

      expect(bindings.dateColumns.value).toBe("created_at");
    });

    it("applies auto_distill", () => {
      const bindings = createBindings();
      usePytorchFormBridge(bindings);
      mountedCallbacks.forEach((cb) => cb());

      const bridge = window.__AIFOLIO_PYTORCH_FORM_BRIDGE__!;
      bridge.applyPatch({ auto_distill: true });

      expect(bindings.autoDistill.value).toBe(true);
    });

    it("applies multiple fields at once", () => {
      const bindings = createBindings();
      usePytorchFormBridge(bindings);
      mountedCallbacks.forEach((cb) => cb());

      const bridge = window.__AIFOLIO_PYTORCH_FORM_BRIDGE__!;
      const result = bridge.applyPatch({
        training_mode: "tabresnet",
        epoch_values: [30, 60],
        batch_sizes: [33, 40],
        hidden_dims: [64, 96],
        dropouts: [0.1, 0.2],
      });

      expect(bindings.trainingMode.value).toBe("tabresnet");
      expect(bindings.epochValues.value).toBe("30,60");
      expect(bindings.batchSizes.value).toBe("33,40");
      expect(bindings.hiddenDims.value).toBe("64,96");
      expect(bindings.dropouts.value).toBe("0.1,0.2");
      expect(result.applied).toHaveLength(5);
      expect(result.skipped).toHaveLength(0);
    });

    it("reports unknown fields as skipped", () => {
      const bindings = createBindings();
      usePytorchFormBridge(bindings);
      mountedCallbacks.forEach((cb) => cb());

      const bridge = window.__AIFOLIO_PYTORCH_FORM_BRIDGE__!;
      const result = bridge.applyPatch({
        training_mode: "mlp_dense",
        unknown_field: "test",
      } as any);

      expect(result.applied).toContain("training_mode");
      expect(result.skipped).toContain("unknown_field");
    });
  });

  describe("startTrainingRuns", () => {
    it("calls onTrain and returns ok", async () => {
      const bindings = createBindings();
      usePytorchFormBridge(bindings);
      mountedCallbacks.forEach((cb) => cb());

      const bridge = window.__AIFOLIO_PYTORCH_FORM_BRIDGE__!;
      const result = await bridge.startTrainingRuns();

      expect(bindings.onTrain).toHaveBeenCalledTimes(1);
      expect(result.status).toBe("ok");
    });

    it("returns error when onTrain throws", async () => {
      const bindings = createBindings();
      bindings.onTrain = vi.fn().mockRejectedValue(new Error("Training failed"));
      usePytorchFormBridge(bindings);
      mountedCallbacks.forEach((cb) => cb());

      const bridge = window.__AIFOLIO_PYTORCH_FORM_BRIDGE__!;
      const result = await bridge.startTrainingRuns();

      expect(result.status).toBe("error");
      expect(result.reason).toBe("Training failed");
    });
  });
});

describe("useTensorflowFormBridge", () => {
  beforeEach(() => {
    mountedCallbacks = [];
    unmountedCallbacks = [];
    delete (window as any).__AIFOLIO_TENSORFLOW_FORM_BRIDGE__;
  });

  afterEach(() => {
    delete (window as any).__AIFOLIO_TENSORFLOW_FORM_BRIDGE__;
  });

  it("registers tensorflow bridge on mount", () => {
    const bindings = createBindings();
    useTensorflowFormBridge(bindings);

    expect(window.__AIFOLIO_TENSORFLOW_FORM_BRIDGE__).toBeUndefined();
    mountedCallbacks.forEach((cb) => cb());
    expect(window.__AIFOLIO_TENSORFLOW_FORM_BRIDGE__).toBeDefined();
  });

  it("removes tensorflow bridge on unmount", () => {
    const bindings = createBindings();
    useTensorflowFormBridge(bindings);

    mountedCallbacks.forEach((cb) => cb());
    expect(window.__AIFOLIO_TENSORFLOW_FORM_BRIDGE__).toBeDefined();

    unmountedCallbacks.forEach((cb) => cb());
    expect(window.__AIFOLIO_TENSORFLOW_FORM_BRIDGE__).toBeUndefined();
  });

  it("applies form patch to tensorflow bridge", () => {
    const bindings = createBindings();
    useTensorflowFormBridge(bindings);
    mountedCallbacks.forEach((cb) => cb());

    const bridge = window.__AIFOLIO_TENSORFLOW_FORM_BRIDGE__!;
    const result = bridge.applyPatch({
      training_mode: "wide_and_deep",
      epoch_values: [20, 40],
      batch_sizes: [32, 64],
    });

    expect(bindings.trainingMode.value).toBe("wide_and_deep");
    expect(bindings.epochValues.value).toBe("20,40");
    expect(bindings.batchSizes.value).toBe("32,64");
    expect(result.applied).toHaveLength(3);
  });

  it("starts tensorflow training runs", async () => {
    const bindings = createBindings();
    useTensorflowFormBridge(bindings);
    mountedCallbacks.forEach((cb) => cb());

    const bridge = window.__AIFOLIO_TENSORFLOW_FORM_BRIDGE__!;
    const result = await bridge.startTrainingRuns();

    expect(bindings.onTrain).toHaveBeenCalledTimes(1);
    expect(result.status).toBe("ok");
  });
});
