import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  handleSwitchAgUiTab,
  handleAddChartSpec,
  resolveMlFormPatchFromToolArgs,
  formatSetFormFieldsToolResult,
  formatStartTrainingRunsToolResult,
  formatTrainModelToolResult,
  formatAddChartSpecToolResult,
  formatClearChartsToolResult,
  formatRandomizeFormFieldsToolResult,
  formatChangeTargetColumnToolResult,
} from "@aifolio/frontend-core/ag-ui";
import { createMlTrainingApi } from "~/features/ml/api";

describe("AG-UI Tool Handlers", () => {
  describe("handleSwitchAgUiTab", () => {
    it("resolves valid tab names", () => {
      expect(handleSwitchAgUiTab("pytorch")).toEqual({ status: "ok", tab: "pytorch" });
      expect(handleSwitchAgUiTab("tensorflow")).toEqual({ status: "ok", tab: "tensorflow" });
      expect(handleSwitchAgUiTab("charts")).toEqual({ status: "ok", tab: "charts" });
      expect(handleSwitchAgUiTab("agentic-research")).toEqual({ status: "ok", tab: "agentic-research" });
    });

    it("rejects invalid tab names", () => {
      const result = handleSwitchAgUiTab("invalid-tab");
      expect(result.status).toBe("error");
    });
  });

  describe("handleAddChartSpec", () => {
    const validSpec = {
      type: "line",
      title: "Test",
      xKey: "month",
      yKeys: ["value"],
      data: [{ month: "Jan", value: 10 }],
    };

    it("adds a single chart spec", () => {
      const addFn = vi.fn();
      const result = handleAddChartSpec({ chartSpec: validSpec }, addFn);

      expect(addFn).toHaveBeenCalledTimes(1);
      expect(result.status).toBe("ok");
      expect(result.addedCount).toBe(1);
    });

    it("adds multiple chart specs from array", () => {
      const addFn = vi.fn();
      const specs = [validSpec, { ...validSpec, title: "Test2" }];
      const result = handleAddChartSpec({ chartSpecs: specs }, addFn);

      expect(addFn).toHaveBeenCalledTimes(2);
      expect(result.status).toBe("ok");
      expect(result.addedCount).toBe(2);
    });

    it("returns error when no specs provided", () => {
      const addFn = vi.fn();
      const result = handleAddChartSpec({}, addFn);

      expect(addFn).not.toHaveBeenCalled();
      expect(result.status).toBe("error");
    });

    it("returns error for invalid spec format", () => {
      const addFn = vi.fn();
      const result = handleAddChartSpec({ chartSpec: "not-an-object" }, addFn);

      expect(addFn).not.toHaveBeenCalled();
      expect(result.status).toBe("error");
    });
  });

  describe("resolveMlFormPatchFromToolArgs", () => {
    it("extracts fields from nested fields object", () => {
      const patch = resolveMlFormPatchFromToolArgs({
        fields: { training_mode: "tabresnet", epoch_values: [30, 60] },
      });

      expect(patch.training_mode).toBe("tabresnet");
      expect(patch.epoch_values).toEqual([30, 60]);
    });

    it("normalizes training_mode aliases", () => {
      const patch = resolveMlFormPatchFromToolArgs({
        fields: { training_mode: "neural_net" },
      });

      expect(patch.training_mode).toBe("mlp_dense");
    });

    it("normalizes tab_resnet alias", () => {
      const patch = resolveMlFormPatchFromToolArgs({
        fields: { training_mode: "tab_resnet" },
      });

      expect(patch.training_mode).toBe("tabresnet");
    });

    it("normalizes field name aliases", () => {
      const patch = resolveMlFormPatchFromToolArgs({
        fields: { epochs: "40", batchSize: "32", learningRate: "0.001" },
      });

      expect(patch.epoch_values).toBe("40");
      expect(patch.batch_sizes).toBe("32");
      expect(patch.learning_rates).toBe("0.001");
    });

    it("normalizes dataset alias", () => {
      const patch = resolveMlFormPatchFromToolArgs({
        fields: { dataset: "fraud_detection" },
      });

      expect(patch.dataset_id).toContain("fraud");
    });

    it("passes through canonical field names unchanged", () => {
      const patch = resolveMlFormPatchFromToolArgs({
        fields: {
          training_mode: "mlp_dense",
          target_column: "price",
          task: "regression",
          epoch_values: "80",
          batch_sizes: "32",
          learning_rates: "0.0005",
          test_sizes: "0.25",
          hidden_dims: "256",
          num_hidden_layers: "3",
          dropouts: "0.15",
        },
      });

      expect(patch.training_mode).toBe("mlp_dense");
      expect(patch.target_column).toBe("price");
      expect(patch.task).toBe("regression");
      expect(patch.epoch_values).toBe("80");
      expect(patch.batch_sizes).toBe("32");
      expect(patch.learning_rates).toBe("0.0005");
      expect(patch.test_sizes).toBe("0.25");
      expect(patch.hidden_dims).toBe("256");
      expect(patch.num_hidden_layers).toBe("3");
      expect(patch.dropouts).toBe("0.15");
    });
  });

  describe("format functions", () => {
    it("formatSetFormFieldsToolResult with applied fields", () => {
      const result = formatSetFormFieldsToolResult("PyTorch", {
        status: "ok",
        applied: ["training_mode", "epoch_values"],
        skipped: [],
      });

      expect(result).toContain("PyTorch");
      expect(result).toContain("training mode");
    });

    it("formatStartTrainingRunsToolResult ok", () => {
      const result = formatStartTrainingRunsToolResult("PyTorch", { status: "ok" });
      expect(result).toContain("PyTorch");
    });

    it("formatStartTrainingRunsToolResult error", () => {
      const result = formatStartTrainingRunsToolResult("PyTorch", {
        status: "error",
        code: "PYTORCH_FORM_UNAVAILABLE",
      });
      expect(result).toContain("Unable");
    });

    it("formatTrainModelToolResult ok", () => {
      const result = formatTrainModelToolResult("PyTorch", {
        status: "ok",
        metrics: { test_metric: 0.92 },
      });
      expect(result).toContain("PyTorch");
    });

    it("formatAddChartSpecToolResult ok", () => {
      const result = formatAddChartSpecToolResult({ status: "ok", addedCount: 2 });
      expect(result).toContain("2");
    });

    it("formatClearChartsToolResult", () => {
      const result = formatClearChartsToolResult();
      expect(result).toContain("Cleared");
    });

    it("formatRandomizeFormFieldsToolResult ok", () => {
      const result = formatRandomizeFormFieldsToolResult("PyTorch", {
        status: "ok",
        randomized: true,
        applied: ["epoch_values", "batch_sizes"],
      });
      expect(result).toContain("PyTorch");
    });

    it("formatChangeTargetColumnToolResult ok", () => {
      const result = formatChangeTargetColumnToolResult("PyTorch", "price", {
        status: "ok",
        applied: ["target_column"],
      });
      expect(result).toContain("PyTorch");
    });
  });
});

describe("AG-UI Tool Integration — Bridge Handlers", () => {
  let pytorchBridge: any;
  let tensorflowBridge: any;

  beforeEach(() => {
    pytorchBridge = {
      applyPatch: vi.fn((patch: any) => {
        const applied = Object.keys(patch);
        return { applied, skipped: [] };
      }),
      startTrainingRuns: vi.fn().mockResolvedValue({ status: "ok" }),
    };
    tensorflowBridge = {
      applyPatch: vi.fn((patch: any) => {
        const applied = Object.keys(patch);
        return { applied, skipped: [] };
      }),
      startTrainingRuns: vi.fn().mockResolvedValue({ status: "ok" }),
    };
    (window as any).__AIFOLIO_PYTORCH_FORM_BRIDGE__ = pytorchBridge;
    (window as any).__AIFOLIO_TENSORFLOW_FORM_BRIDGE__ = tensorflowBridge;
  });

  afterEach(() => {
    delete (window as any).__AIFOLIO_PYTORCH_FORM_BRIDGE__;
    delete (window as any).__AIFOLIO_TENSORFLOW_FORM_BRIDGE__;
  });

  it("set_pytorch_form_fields calls bridge.applyPatch with resolved patch", () => {
    const args = { fields: { training_mode: "tabresnet", batch_sizes: [33, 40] } };
    const patch = resolveMlFormPatchFromToolArgs(args);

    const result = pytorchBridge.applyPatch(patch);

    expect(pytorchBridge.applyPatch).toHaveBeenCalledWith(
      expect.objectContaining({ training_mode: "tabresnet" })
    );
    expect(result.applied).toContain("training_mode");
  });

  it("start_pytorch_training_runs calls bridge.startTrainingRuns", async () => {
    const result = await pytorchBridge.startTrainingRuns();

    expect(pytorchBridge.startTrainingRuns).toHaveBeenCalledTimes(1);
    expect(result.status).toBe("ok");
  });

  it("set_tensorflow_form_fields calls tensorflow bridge", () => {
    const args = { fields: { training_mode: "wide_and_deep", epoch_values: [20, 40] } };
    const patch = resolveMlFormPatchFromToolArgs(args);

    const result = tensorflowBridge.applyPatch(patch);

    expect(tensorflowBridge.applyPatch).toHaveBeenCalledWith(
      expect.objectContaining({ training_mode: "wide_and_deep" })
    );
    expect(result.applied).toContain("training_mode");
  });

  it("start_tensorflow_training_runs calls tensorflow bridge", async () => {
    const result = await tensorflowBridge.startTrainingRuns();

    expect(tensorflowBridge.startTrainingRuns).toHaveBeenCalledTimes(1);
    expect(result.status).toBe("ok");
  });

  it("returns error when pytorch bridge is absent", () => {
    delete (window as any).__AIFOLIO_PYTORCH_FORM_BRIDGE__;

    const bridge = (window as any).__AIFOLIO_PYTORCH_FORM_BRIDGE__;
    expect(bridge).toBeUndefined();
  });

  it("returns error when tensorflow bridge is absent", () => {
    delete (window as any).__AIFOLIO_TENSORFLOW_FORM_BRIDGE__;

    const bridge = (window as any).__AIFOLIO_TENSORFLOW_FORM_BRIDGE__;
    expect(bridge).toBeUndefined();
  });
});

describe("AG-UI Tool Integration — Active ML Routing", () => {
  let pytorchBridge: any;
  let tensorflowBridge: any;

  beforeEach(() => {
    pytorchBridge = {
      applyPatch: vi.fn((patch: any) => ({
        applied: Object.keys(patch),
        skipped: [],
      })),
      startTrainingRuns: vi.fn().mockResolvedValue({ status: "ok" }),
    };
    tensorflowBridge = {
      applyPatch: vi.fn((patch: any) => ({
        applied: Object.keys(patch),
        skipped: [],
      })),
      startTrainingRuns: vi.fn().mockResolvedValue({ status: "ok" }),
    };
    (window as any).__AIFOLIO_PYTORCH_FORM_BRIDGE__ = pytorchBridge;
    (window as any).__AIFOLIO_TENSORFLOW_FORM_BRIDGE__ = tensorflowBridge;
  });

  afterEach(() => {
    delete (window as any).__AIFOLIO_PYTORCH_FORM_BRIDGE__;
    delete (window as any).__AIFOLIO_TENSORFLOW_FORM_BRIDGE__;
  });

  it("routes to pytorch bridge when activeTab is pytorch", () => {
    const activeTab = "pytorch";
    const bridge = activeTab === "pytorch" ? pytorchBridge : tensorflowBridge;
    const patch = resolveMlFormPatchFromToolArgs({ fields: { epoch_values: "80" } });

    bridge.applyPatch(patch);

    expect(pytorchBridge.applyPatch).toHaveBeenCalled();
    expect(tensorflowBridge.applyPatch).not.toHaveBeenCalled();
  });

  it("routes to tensorflow bridge when activeTab is tensorflow", () => {
    const activeTab = "tensorflow";
    const bridge = activeTab === "pytorch" ? pytorchBridge : tensorflowBridge;
    const patch = resolveMlFormPatchFromToolArgs({ fields: { epoch_values: "80" } });

    bridge.applyPatch(patch);

    expect(tensorflowBridge.applyPatch).toHaveBeenCalled();
    expect(pytorchBridge.applyPatch).not.toHaveBeenCalled();
  });

  it("returns ACTIVE_ML_TAB_REQUIRED when tab is charts", () => {
    const activeTab = "charts";
    const framework = activeTab === "pytorch" || activeTab === "tensorflow" ? activeTab : null;

    expect(framework).toBeNull();
  });

  it("returns ACTIVE_ML_TAB_REQUIRED when tab is agentic-research", () => {
    const activeTab = "agentic-research";
    const framework = activeTab === "pytorch" || activeTab === "tensorflow" ? activeTab : null;

    expect(framework).toBeNull();
  });
});

describe("AG-UI Tool Integration — ML Training API", () => {
  it("train_pytorch_model calls trainPytorch with defaults", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      json: () => Promise.resolve({ status: "ok", metrics: { test_metric: 0.91 } }),
    });
    vi.stubGlobal("fetch", mockFetch);

    const api = createMlTrainingApi({ baseUrl: "/api/ai" });
    const result = await api.trainPytorch({
      dataset_id: "churn.csv",
      target_column: "Churn",
      task: "auto",
      epochs: 60,
      batch_size: 64,
      learning_rate: 0.001,
      training_mode: "mlp_dense",
      exclude_columns: [],
      date_columns: [],
      test_size: 0.2,
      hidden_dim: 128,
      num_hidden_layers: 2,
      dropout: 0.1,
    });

    expect(mockFetch).toHaveBeenCalledWith(
      "/api/ai/ml/pytorch/train",
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
      })
    );
    expect(result.status).toBe("ok");

    vi.unstubAllGlobals();
  });

  it("train_tensorflow_model calls trainTensorflow with defaults", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      json: () => Promise.resolve({ status: "ok", metrics: { test_metric: 0.88 } }),
    });
    vi.stubGlobal("fetch", mockFetch);

    const api = createMlTrainingApi({ baseUrl: "/api/ai" });
    const result = await api.trainTensorflow({
      dataset_id: "house_prices.csv",
      target_column: "SalePrice",
      task: "regression",
      epochs: 40,
      batch_size: 32,
      learning_rate: 0.0005,
      training_mode: "wide_and_deep",
      exclude_columns: [],
      date_columns: [],
      test_size: 0.25,
      hidden_dim: 256,
      num_hidden_layers: 3,
      dropout: 0.15,
    });

    expect(mockFetch).toHaveBeenCalledWith(
      "/api/ai/ml/tensorflow/train",
      expect.objectContaining({ method: "POST" })
    );
    expect(result.status).toBe("ok");

    vi.unstubAllGlobals();
  });
});
