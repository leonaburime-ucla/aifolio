import { describe, expect, it, vi } from "vitest";
import {
  createReloadSweepValuesHandler,
  createToggleRunSweepHandler,
  handleApplyOptimalParams,
  handleCopyTrainingRuns,
  handleFindOptimalParams,
} from "@/features/ml/typescript/react/hooks/logic/trainingShared.logic";

function buildNumericUi() {
  return {
    epochValuesInput: "60",
    batchSizesInput: "64",
    learningRatesInput: "0.001",
    testSizesInput: "0.2",
    hiddenDimsInput: "128",
    numHiddenLayersInput: "2",
    dropoutsInput: "0.1",
    savedSweepInputs: null,
    savedNumericInputs: null,
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

describe("trainingShared.logic", () => {
  it("toggles run sweep on/off with snapshot handling", () => {
    const ui = buildNumericUi();
    const toggle = createToggleRunSweepHandler({ ui, defaultEpochs: 42 });

    toggle(true);
    expect(ui.setSavedNumericInputs).toHaveBeenCalledTimes(1);
    expect(ui.setSavedSweepInputs).toHaveBeenCalledTimes(1);
    expect(ui.setRunSweepEnabled).toHaveBeenCalledWith(true);

    toggle(false);
    expect(ui.setRunSweepEnabled).toHaveBeenCalledWith(false);
  });

  it("restores default numeric values when sweep is disabled without saved snapshot", () => {
    const ui = buildNumericUi();
    const toggle = createToggleRunSweepHandler({ ui, defaultEpochs: 99 });
    toggle(false);

    expect(ui.setEpochValuesInput).toHaveBeenCalledWith("99");
    expect(ui.setBatchSizesInput).toHaveBeenCalledWith("64");
    expect(ui.setDropoutsInput).toHaveBeenCalledWith("0.1");
    expect(ui.setRunSweepEnabled).toHaveBeenCalledWith(false);
  });

  it("reloads sweep values", () => {
    const ui = buildNumericUi();
    const reload = createReloadSweepValuesHandler({
      ui: {
        setSavedSweepInputs: ui.setSavedSweepInputs,
        setEpochValuesInput: ui.setEpochValuesInput,
        setBatchSizesInput: ui.setBatchSizesInput,
        setLearningRatesInput: ui.setLearningRatesInput,
        setTestSizesInput: ui.setTestSizesInput,
        setHiddenDimsInput: ui.setHiddenDimsInput,
        setNumHiddenLayersInput: ui.setNumHiddenLayersInput,
        setDropoutsInput: ui.setDropoutsInput,
      },
    });

    reload();
    expect(ui.setSavedSweepInputs).toHaveBeenCalledTimes(1);
    expect(ui.setEpochValuesInput).toHaveBeenCalledTimes(1);
  });

  it("sets helpful status when optimizer lacks enough runs", () => {
    const ui = {
      setPendingOptimalParams: vi.fn(),
      setPendingOptimalPrediction: vi.fn(),
      setIsOptimalModalOpen: vi.fn(),
      setOptimizerStatus: vi.fn(),
    };
    const runtime = {
      schedule: vi.fn((callback: () => void) => callback()),
      writeClipboardText: vi.fn(async () => undefined),
    };

    handleFindOptimalParams(
      { trainingRuns: [{ result: "completed" }], ui },
      { runtime }
    );

    expect(ui.setOptimizerStatus).toHaveBeenCalledWith(
      "Need at least 5 completed runs for the specific algorithm."
    );
  });

  it("sets optimal modal data when suggestion is available", () => {
    const ui = {
      setPendingOptimalParams: vi.fn(),
      setPendingOptimalPrediction: vi.fn(),
      setIsOptimalModalOpen: vi.fn(),
      setOptimizerStatus: vi.fn(),
    };
    const runtime = {
      schedule: vi.fn((callback: () => void) => callback()),
      writeClipboardText: vi.fn(async () => undefined),
    };
    const rows = [
      { result: "completed", metric_name: "accuracy", metric_score: 0.7, epochs: 10, learning_rate: 0.001, test_size: 0.2, batch_size: 32, hidden_dim: 64, num_hidden_layers: 2, dropout: 0.1 },
      { result: "completed", metric_name: "accuracy", metric_score: 0.71, epochs: 12, learning_rate: 0.0011, test_size: 0.2, batch_size: 32, hidden_dim: 64, num_hidden_layers: 2, dropout: 0.11 },
      { result: "completed", metric_name: "accuracy", metric_score: 0.72, epochs: 14, learning_rate: 0.0012, test_size: 0.2, batch_size: 32, hidden_dim: 64, num_hidden_layers: 2, dropout: 0.12 },
      { result: "completed", metric_name: "accuracy", metric_score: 0.73, epochs: 16, learning_rate: 0.0013, test_size: 0.2, batch_size: 32, hidden_dim: 64, num_hidden_layers: 2, dropout: 0.13 },
      { result: "completed", metric_name: "accuracy", metric_score: 0.74, epochs: 18, learning_rate: 0.0014, test_size: 0.2, batch_size: 32, hidden_dim: 64, num_hidden_layers: 2, dropout: 0.14 },
    ];

    handleFindOptimalParams({ trainingRuns: rows, ui }, { runtime });

    expect(ui.setPendingOptimalParams).toHaveBeenCalledTimes(1);
    expect(ui.setPendingOptimalPrediction).toHaveBeenCalledTimes(1);
    expect(ui.setIsOptimalModalOpen).toHaveBeenCalledWith(true);
  });

  it("applies pending optimal params to numeric inputs", () => {
    const ui = {
      pendingOptimalParams: {
        epochs: 100,
        learning_rate: 0.0012345,
        test_size: 0.23456,
        batch_size: 64,
        hidden_dim: 128,
        num_hidden_layers: 3,
        dropout: 0.25,
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
    const runtime = {
      schedule: vi.fn((callback: () => void) => callback()),
      writeClipboardText: vi.fn(async () => undefined),
    };

    handleApplyOptimalParams({ ui }, { runtime });

    expect(ui.setEpochValuesInput).toHaveBeenCalledWith("100");
    expect(ui.setBatchSizesInput).toHaveBeenCalledWith("64");
    expect(ui.setIsOptimalModalOpen).toHaveBeenCalledWith(false);
  });

  it("returns early when apply called without pending params", () => {
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

    handleApplyOptimalParams({ ui });
    expect(ui.setEpochValuesInput).not.toHaveBeenCalled();
  });

  it("copies training runs and sets copied status", async () => {
    const runtime = {
      schedule: vi.fn((callback: () => void) => callback()),
      writeClipboardText: vi.fn(async () => undefined),
    };
    const setCopyRunsStatus = vi.fn();

    await handleCopyTrainingRuns(
      {
        trainingRuns: [{ completed_at: "now", result: "completed", distill_action: "" }],
        setCopyRunsStatus,
      },
      { runtime }
    );

    expect(runtime.writeClipboardText).toHaveBeenCalledTimes(1);
    expect(setCopyRunsStatus).toHaveBeenCalledWith("Copied");
  });

  it("no-ops copy for empty runs and sets failure status on clipboard error", async () => {
    const setCopyRunsStatus = vi.fn();

    await handleCopyTrainingRuns({ trainingRuns: [], setCopyRunsStatus });
    expect(setCopyRunsStatus).not.toHaveBeenCalled();

    const runtime = {
      schedule: vi.fn((callback: () => void) => callback()),
      writeClipboardText: vi.fn(async () => {
        throw new Error("clipboard down");
      }),
    };

    await handleCopyTrainingRuns(
      {
        trainingRuns: [{ completed_at: "now", result: "completed" }],
        setCopyRunsStatus,
      },
      { runtime }
    );

    expect(setCopyRunsStatus).toHaveBeenCalledWith("Copy failed");
  });

  it("handles non-Error clipboard failures", async () => {
    const runtime = {
      schedule: vi.fn((callback: () => void) => callback()),
      writeClipboardText: vi.fn(async () => {
        throw "bad";
      }),
    };
    const setCopyRunsStatus = vi.fn();

    await handleCopyTrainingRuns(
      {
        trainingRuns: [{ completed_at: "now", result: "completed" }],
        setCopyRunsStatus,
      },
      { runtime }
    );

    expect(setCopyRunsStatus).toHaveBeenCalledWith("Copy failed");
  });
});
