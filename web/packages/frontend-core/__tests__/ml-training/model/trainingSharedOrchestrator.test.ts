import { describe, expect, it, vi } from "vitest";
import {
  createToggleRunSweepHandler,
  createReloadSweepValuesHandler,
  handleFindOptimalParams,
  handleApplyOptimalParams,
  handleCopyTrainingRuns,
} from "../../../src/ml-training/model/trainingShared";
import type { TrainingSharedRuntime } from "@aifolio/contracts/entities/ml-training";

function createMockRuntime(): TrainingSharedRuntime {
  return {
    schedule: vi.fn(),
    writeClipboardText: vi.fn().mockResolvedValue(undefined),
  };
}

function createNumericInputUi() {
  return {
    epochValuesInput: "10",
    batchSizesInput: "32",
    learningRatesInput: "0.001",
    testSizesInput: "0.2",
    hiddenDimsInput: "128",
    numHiddenLayersInput: "2",
    dropoutsInput: "0.1",
    savedNumericInputs: null as Record<string, string> | null,
    savedSweepInputs: null as Record<string, string> | null,
    setSavedNumericInputs: vi.fn(),
    setSavedSweepInputs: vi.fn(),
    setEpochValuesInput: vi.fn(),
    setBatchSizesInput: vi.fn(),
    setLearningRatesInput: vi.fn(),
    setTestSizesInput: vi.fn(),
    setHiddenDimsInput: vi.fn(),
    setNumHiddenLayersInput: vi.fn(),
    setDropoutsInput: vi.fn(),
    setRunSweepEnabled: vi.fn(),
  };
}

function makeCompletedRun(overrides: Record<string, unknown> = {}) {
  return {
    result: "completed",
    metric_name: "accuracy",
    metric_score: "0.8",
    epochs: "60",
    learning_rate: "0.001",
    test_size: "0.2",
    batch_size: "64",
    hidden_dim: "128",
    num_hidden_layers: "2",
    dropout: "0.1",
    ...overrides,
  };
}

describe("createToggleRunSweepHandler", () => {
  it("saves current inputs and applies sweep inputs when toggling on", () => {
    const ui = createNumericInputUi();
    const toggle = createToggleRunSweepHandler({ ui, defaultEpochs: 50 });
    toggle(true);

    expect(ui.setSavedNumericInputs).toHaveBeenCalledWith({
      epochValuesInput: "10",
      batchSizesInput: "32",
      learningRatesInput: "0.001",
      testSizesInput: "0.2",
      hiddenDimsInput: "128",
      numHiddenLayersInput: "2",
      dropoutsInput: "0.1",
    });
    expect(ui.setRunSweepEnabled).toHaveBeenCalledWith(true);
    expect(ui.setSavedSweepInputs).toHaveBeenCalled();
  });

  it("restores saved numeric inputs when toggling off with saved state", () => {
    const ui = createNumericInputUi();
    ui.savedNumericInputs = {
      epochValuesInput: "20",
      batchSizesInput: "64",
      learningRatesInput: "0.01",
      testSizesInput: "0.3",
      hiddenDimsInput: "256",
      numHiddenLayersInput: "3",
      dropoutsInput: "0.2",
    };
    const toggle = createToggleRunSweepHandler({ ui, defaultEpochs: 50 });
    toggle(false);

    expect(ui.setEpochValuesInput).toHaveBeenCalledWith("20");
    expect(ui.setBatchSizesInput).toHaveBeenCalledWith("64");
    expect(ui.setRunSweepEnabled).toHaveBeenCalledWith(false);
  });

  it("applies default values when toggling off without saved state", () => {
    const ui = createNumericInputUi();
    const toggle = createToggleRunSweepHandler({ ui, defaultEpochs: 50 });
    toggle(false);

    expect(ui.setEpochValuesInput).toHaveBeenCalledWith("50");
    expect(ui.setBatchSizesInput).toHaveBeenCalledWith("64");
    expect(ui.setLearningRatesInput).toHaveBeenCalledWith("0.001");
    expect(ui.setRunSweepEnabled).toHaveBeenCalledWith(false);
  });
});

describe("createReloadSweepValuesHandler", () => {
  it("regenerates and applies random sweep values", () => {
    const ui = createNumericInputUi();
    const reload = createReloadSweepValuesHandler({ ui });
    reload();

    expect(ui.setSavedSweepInputs).toHaveBeenCalled();
    expect(ui.setEpochValuesInput).toHaveBeenCalled();
    expect(ui.setBatchSizesInput).toHaveBeenCalled();
  });
});

describe("handleFindOptimalParams", () => {
  it("sets optimizer status when insufficient runs", () => {
    const runtime = createMockRuntime();
    const ui = {
      setOptimizerStatus: vi.fn(),
      setPendingOptimalParams: vi.fn(),
      setPendingOptimalPrediction: vi.fn(),
      setIsOptimalModalOpen: vi.fn(),
    };

    handleFindOptimalParams({ trainingRuns: [], ui }, { runtime });

    expect(ui.setOptimizerStatus).toHaveBeenCalledWith(
      "Need at least 5 completed runs for the specific algorithm."
    );
    expect(runtime.schedule).toHaveBeenCalled();
  });

  it("opens the optimizer modal when enough completed runs exist", () => {
    const runtime = createMockRuntime();
    const ui = {
      setOptimizerStatus: vi.fn(),
      setPendingOptimalParams: vi.fn(),
      setPendingOptimalPrediction: vi.fn(),
      setIsOptimalModalOpen: vi.fn(),
    };

    handleFindOptimalParams({
      trainingRuns: [
        makeCompletedRun({ metric_score: "0.71", epochs: "40" }),
        makeCompletedRun({ metric_score: "0.74", epochs: "50" }),
        makeCompletedRun({ metric_score: "0.78", epochs: "55" }),
        makeCompletedRun({ metric_score: "0.81", epochs: "60" }),
        makeCompletedRun({ metric_score: "0.85", epochs: "70" }),
      ] as never[],
      ui,
    }, { runtime });

    expect(ui.setPendingOptimalParams).toHaveBeenCalledWith(
      expect.objectContaining({
        epochs: expect.any(Number),
        learning_rate: expect.any(Number),
      })
    );
    expect(ui.setPendingOptimalPrediction).toHaveBeenCalledWith({
      metricName: "accuracy",
      metricValue: expect.any(Number),
    });
    expect(ui.setIsOptimalModalOpen).toHaveBeenCalledWith(true);
    expect(ui.setOptimizerStatus).toHaveBeenCalledWith("Suggestion generated from 5 runs.");
    expect(runtime.schedule).toHaveBeenCalled();
  });
});

describe("handleApplyOptimalParams", () => {
  it("does nothing when pendingOptimalParams is null", () => {
    const runtime = createMockRuntime();
    const ui = {
      pendingOptimalParams: null,
      setEpochValuesInput: vi.fn(),
      setLearningRatesInput: vi.fn(),
      setTestSizesInput: vi.fn(),
      setBatchSizesInput: vi.fn(),
      setHiddenDimsInput: vi.fn(),
      setNumHiddenLayersInput: vi.fn(),
      setDropoutsInput: vi.fn(),
      setRunSweepEnabled: vi.fn(),
      setIsOptimalModalOpen: vi.fn(),
      setPendingOptimalPrediction: vi.fn(),
      setOptimizerStatus: vi.fn(),
    };

    handleApplyOptimalParams({ ui }, { runtime });

    expect(ui.setEpochValuesInput).not.toHaveBeenCalled();
  });

  it("applies optimal params to form state and closes modal", () => {
    const runtime = createMockRuntime();
    const ui = {
      pendingOptimalParams: {
        epochs: 25,
        learning_rate: 0.0012345,
        test_size: 0.25,
        batch_size: 64,
        hidden_dim: 256,
        num_hidden_layers: 3,
        dropout: 0.15,
      },
      setEpochValuesInput: vi.fn(),
      setLearningRatesInput: vi.fn(),
      setTestSizesInput: vi.fn(),
      setBatchSizesInput: vi.fn(),
      setHiddenDimsInput: vi.fn(),
      setNumHiddenLayersInput: vi.fn(),
      setDropoutsInput: vi.fn(),
      setRunSweepEnabled: vi.fn(),
      setIsOptimalModalOpen: vi.fn(),
      setPendingOptimalPrediction: vi.fn(),
      setOptimizerStatus: vi.fn(),
    };

    handleApplyOptimalParams({ ui }, { runtime });

    expect(ui.setEpochValuesInput).toHaveBeenCalledWith("25");
    expect(ui.setBatchSizesInput).toHaveBeenCalledWith("64");
    expect(ui.setHiddenDimsInput).toHaveBeenCalledWith("256");
    expect(ui.setNumHiddenLayersInput).toHaveBeenCalledWith("3");
    expect(ui.setRunSweepEnabled).toHaveBeenCalledWith(false);
    expect(ui.setIsOptimalModalOpen).toHaveBeenCalledWith(false);
    expect(ui.setPendingOptimalPrediction).toHaveBeenCalledWith(null);
    expect(ui.setOptimizerStatus).toHaveBeenCalledWith("Updated table with suggested values.");
    expect(runtime.schedule).toHaveBeenCalled();
  });
});

describe("handleCopyTrainingRuns", () => {
  it("does nothing for empty runs array", async () => {
    const runtime = createMockRuntime();
    const setCopyRunsStatus = vi.fn();

    await handleCopyTrainingRuns({ trainingRuns: [], setCopyRunsStatus }, { runtime });

    expect(runtime.writeClipboardText).not.toHaveBeenCalled();
    expect(setCopyRunsStatus).not.toHaveBeenCalled();
  });

  it("copies runs as TSV and sets status to Copied", async () => {
    const runtime = createMockRuntime();
    const setCopyRunsStatus = vi.fn();
    const fakeRun = {
      run_id: "r1",
      dataset_id: "d.csv",
      training_mode: "mlp_dense",
      target_column: "y",
      task: "regression",
      epochs: 10,
      batch_size: 32,
      learning_rate: 0.001,
      test_size: 0.2,
      hidden_dim: 128,
      num_hidden_layers: 2,
      dropout: 0.1,
      train_loss: 0.5,
      test_loss: 0.6,
      r2_score: 0.9,
      accuracy: null,
      precision: null,
      recall: null,
      f1_score: null,
      status: "completed",
      distill_action: "",
    };

    await handleCopyTrainingRuns(
      { trainingRuns: [fakeRun as never], setCopyRunsStatus },
      { runtime }
    );

    expect(runtime.writeClipboardText).toHaveBeenCalled();
    expect(runtime.writeClipboardText).toHaveBeenCalledWith(
      expect.stringContaining("Not Available")
    );
    expect(setCopyRunsStatus).toHaveBeenCalledWith("Copied");
    expect(runtime.schedule).toHaveBeenCalled();
  });

  it("preserves nonblank distillation action text when copying runs", async () => {
    const runtime = createMockRuntime();
    const setCopyRunsStatus = vi.fn();
    const fakeRun = {
      run_id: "r2",
      dataset_id: "d.csv",
      training_mode: "mlp_dense",
      target_column: "y",
      task: "regression",
      epochs: 10,
      batch_size: 32,
      learning_rate: 0.001,
      test_size: 0.2,
      hidden_dim: 128,
      num_hidden_layers: 2,
      dropout: 0.1,
      train_loss: 0.5,
      test_loss: 0.6,
      r2_score: 0.9,
      accuracy: null,
      precision: null,
      recall: null,
      f1_score: null,
      status: "completed",
      distill_action: "Distill",
    };

    await handleCopyTrainingRuns(
      { trainingRuns: [fakeRun as never], setCopyRunsStatus },
      { runtime }
    );

    expect(runtime.writeClipboardText).toHaveBeenCalledWith(
      expect.stringContaining("Distill")
    );
  });

  it("sets Copy failed status when clipboard write throws", async () => {
    const runtime = createMockRuntime();
    (runtime.writeClipboardText as ReturnType<typeof vi.fn>).mockRejectedValue(
      new Error("Permission denied")
    );
    const setCopyRunsStatus = vi.fn();
    const fakeRun = {
      run_id: "r1",
      dataset_id: "d.csv",
      training_mode: "mlp_dense",
      target_column: "y",
      task: "regression",
      epochs: 10,
      batch_size: 32,
      learning_rate: 0.001,
      test_size: 0.2,
      hidden_dim: 128,
      num_hidden_layers: 2,
      dropout: 0.1,
      train_loss: 0.5,
      test_loss: 0.6,
      r2_score: 0.9,
      accuracy: null,
      precision: null,
      recall: null,
      f1_score: null,
      status: "completed",
      distill_action: "",
    };

    await handleCopyTrainingRuns(
      { trainingRuns: [fakeRun as never], setCopyRunsStatus },
      { runtime }
    );

    expect(setCopyRunsStatus).toHaveBeenCalledWith("Copy failed");
  });
});
