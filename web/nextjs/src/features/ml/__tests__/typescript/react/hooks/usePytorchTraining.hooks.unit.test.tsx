import { act, renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { toast } from "react-hot-toast";
import {
  usePytorchLogic,
  usePytorchUiState,
} from "@/features/ml/typescript/react/hooks/usePytorchTraining.hooks";
import type { PytorchLogicArgs } from "@/features/ml/__types__/typescript/react/hooks/pytorchTraining.types";

vi.mock("react-hot-toast", () => ({
  toast: {
    success: vi.fn(),
    error: vi.fn(),
  },
}));

function createUi(
  overrides: Record<string, unknown> = {}
): PytorchLogicArgs["ui"] {
  return {
    trainingMode: "mlp_dense",
    setTrainingMode: vi.fn(),
    targetColumn: "Churn",
    setTargetColumn: vi.fn(),
    excludeColumnsInput: null,
    setExcludeColumnsInput: vi.fn(),
    dateColumnsInput: null,
    setDateColumnsInput: vi.fn(),
    task: "classification",
    setTask: vi.fn(),
    epochValuesInput: "60",
    setEpochValuesInput: vi.fn(),
    testSizesInput: "0.2",
    setTestSizesInput: vi.fn(),
    learningRatesInput: "0.001",
    setLearningRatesInput: vi.fn(),
    batchSizesInput: "64",
    setBatchSizesInput: vi.fn(),
    hiddenDimsInput: "128",
    setHiddenDimsInput: vi.fn(),
    numHiddenLayersInput: "2",
    setNumHiddenLayersInput: vi.fn(),
    dropoutsInput: "0.1",
    setDropoutsInput: vi.fn(),
    runSweepEnabled: false,
    setRunSweepEnabled: vi.fn(),
    savedNumericInputs: null,
    setSavedNumericInputs: vi.fn(),
    savedSweepInputs: null,
    setSavedSweepInputs: vi.fn(),
    isTraining: false,
    setIsTraining: vi.fn(),
    isDistilling: false,
    setIsDistilling: vi.fn(),
    autoDistillEnabled: false,
    setAutoDistillEnabled: vi.fn(),
    trainingProgress: { current: 0, total: 0 },
    setTrainingProgress: vi.fn(),
    trainingError: null,
    setTrainingError: vi.fn(),
    copyRunsStatus: null,
    setCopyRunsStatus: vi.fn(),
    optimizerStatus: null,
    setOptimizerStatus: vi.fn(),
    distillStatus: null,
    setDistillStatus: vi.fn(),
    saveDistilledModel: false,
    setSaveDistilledModel: vi.fn(),
    isOptimalModalOpen: false,
    setIsOptimalModalOpen: vi.fn(),
    pendingOptimalParams: null,
    setPendingOptimalParams: vi.fn(),
    pendingOptimalPrediction: null,
    setPendingOptimalPrediction: vi.fn(),
    isDistillMetricsModalOpen: false,
    setIsDistillMetricsModalOpen: vi.fn(),
    distillMetrics: null,
    setDistillMetrics: vi.fn(),
    distillModelId: null,
    setDistillModelId: vi.fn(),
    distillModelPath: null,
    setDistillModelPath: vi.fn(),
    distillComparison: null,
    setDistillComparison: vi.fn(),
    ...overrides,
  } as unknown as PytorchLogicArgs["ui"];
}

function createDataset(
  overrides: Record<string, unknown> = {}
): PytorchLogicArgs["dataset"] {
  return {
    selectedDatasetId: "customer_churn_telco.csv",
    setSelectedDatasetId: vi.fn(),
    tableColumns: ["Churn", "Age"],
    ...overrides,
  } as unknown as PytorchLogicArgs["dataset"];
}

describe("usePytorchTraining.hooks", () => {
  it("provides default ui mode", () => {
    const { result } = renderHook(() => usePytorchUiState());
    expect(result.current.trainingMode).toBe("mlp_dense");

    act(() => {
      result.current.setTrainingMode("tabresnet");
    });
    expect(result.current.trainingMode).toBe("tabresnet");
  });

  it("resets values on dataset change", () => {
    const ui = createUi();
    const dataset = createDataset();
    const { result } = renderHook(() =>
      usePytorchLogic({
        dataset,
        trainingRuns: [],
        prependTrainingRun: vi.fn(),
        ui,
        runTraining: vi.fn(),
        runDistillation: vi.fn(),
      })
    );

    act(() => {
      result.current.onDatasetChange("house_prices_ames.csv");
    });

    expect(dataset.setSelectedDatasetId).toHaveBeenCalledWith("house_prices_ames.csv");
    expect(ui.setTargetColumn).toHaveBeenCalledWith("SalePrice");
    expect(ui.setTask).toHaveBeenCalledWith("regression");
  });

  it("blocks training when setup is invalid", async () => {
    const ui = createUi();
    const dataset = createDataset({ selectedDatasetId: null, tableColumns: [] });
    const runTraining = vi.fn();
    const { result } = renderHook(() =>
      usePytorchLogic({
        dataset,
        trainingRuns: [],
        prependTrainingRun: vi.fn(),
        ui,
        runTraining,
        runDistillation: vi.fn(),
      })
    );

    await act(async () => {
      await result.current.onTrainClick();
    });

    expect(runTraining).not.toHaveBeenCalled();
    expect(ui.setTrainingError).toHaveBeenCalledTimes(1);
  });

  it("handles training outcomes, stop requests, and copy", async () => {
    const notifySuccess = vi.fn();
    const notifyError = vi.fn();
    const schedule = vi.fn((callback: () => void) => callback());
    const writeClipboardText = vi.fn(async () => undefined);
    const ui = createUi({ autoDistillEnabled: true, trainingMode: "unsupported_mode" });
    const dataset = createDataset();
    const runTraining = vi.fn(async () => ({
      stopped: false,
      completed: 2,
      total: 2,
      completedTeacherRuns: [{ run_id: "teacher-1", model_id: "m1", result: "completed" }],
      failedRuns: 1,
      firstFailureMessage: "run failed",
    }));

    const { result } = renderHook(() =>
      usePytorchLogic({
        dataset,
        trainingRuns: [{ run_id: "r1", result: "completed", metric_name: "accuracy", metric_score: 0.9, training_mode: "mlp_dense" }],
        prependTrainingRun: vi.fn(),
        ui,
        runTraining,
        runDistillation: vi.fn(),
        runtime: { notifySuccess, notifyError, schedule, writeClipboardText },
      })
    );

    act(() => {
      result.current.onStopTrainingRuns();
    });

    await act(async () => {
      await result.current.onTrainClick();
      await result.current.onCopyTrainingRuns();
    });

    expect(runTraining).toHaveBeenCalledTimes(1);
    expect(notifyError).toHaveBeenCalled();
    expect(notifySuccess).toHaveBeenCalled();
    expect(ui.setDistillStatus).toHaveBeenCalledWith(
      "Auto-distill skipped: 'unsupported_mode' distillation is not supported yet."
    );
    expect(writeClipboardText).toHaveBeenCalledTimes(1);
  });

  it("runs distillation and supports fallback viewing", async () => {
    const ui = createUi();
    const prependTrainingRun = vi.fn();
    const dataset = createDataset();
    const runDistillation = vi.fn(async () => ({
      status: "ok",
      metrics: { test_metric_name: "accuracy", test_metric_value: 0.91 },
      modelId: "student-1",
      modelPath: "/tmp/student-1",
      runId: "run-student-1",
      teacherModelSizeBytes: 1000,
      studentModelSizeBytes: 500,
      teacherInputDim: 10,
      teacherOutputDim: 2,
      studentInputDim: 10,
      studentOutputDim: 2,
      sizeSavedBytes: 500,
      sizeSavedPercent: 50,
      teacherParamCount: 1200,
      studentParamCount: 600,
      paramSavedCount: 600,
      paramSavedPercent: 50,
      distilledRun: {
        result: "distilled",
        run_id: "run-student-1",
        model_id: "student-1",
        model_path: "/tmp/student-1",
        metric_name: "accuracy",
        metric_score: 0.91,
        hidden_dim: 32,
        num_hidden_layers: 2,
        training_mode: "mlp_dense",
      },
    }));

    const { result } = renderHook(() =>
      usePytorchLogic({
        dataset,
        trainingRuns: [],
        prependTrainingRun,
        ui,
        runTraining: vi.fn(),
        runDistillation,
      })
    );

    const teacherRun = {
      result: "completed",
      run_id: "teacher-1",
      model_id: "teacher-model",
      model_path: "/tmp/teacher",
      metric_name: "accuracy",
      metric_score: 0.88,
      hidden_dim: 128,
      num_hidden_layers: 3,
      dropout: 0.1,
      epochs: 60,
      batch_size: 64,
      learning_rate: 0.001,
      test_size: 0.2,
      training_mode: "mlp_dense",
    };

    await act(async () => {
      await result.current.onDistillFromRun(teacherRun);
    });

    expect(runDistillation).toHaveBeenCalledTimes(1);
    expect(prependTrainingRun).toHaveBeenCalledTimes(1);
    expect(ui.setIsDistillMetricsModalOpen).toHaveBeenCalledWith(true);

    act(() => {
      result.current.onSeeDistilledFromRun({ run_id: "teacher-1", result: "completed" });
    });
    expect(ui.setDistillComparison).toHaveBeenCalled();

    act(() => {
      result.current.onSeeDistilledFromRun({ run_id: "missing", result: "completed" });
    });
    expect(ui.setTrainingError).toHaveBeenCalledWith("No distilled result found yet for this teacher run.");
  });

  it("uses fallback distilled rows when no in-memory snapshot exists", () => {
    const ui = createUi();
    const { result } = renderHook(() =>
      usePytorchLogic({
        dataset: createDataset(),
        trainingRuns: [
          {
            result: "distilled",
            teacher_ref_key: "teacher-fallback",
            run_id: "distilled-1",
            model_id: "model-distilled-1",
            model_path: "/tmp/distilled-1",
            metric_name: "accuracy",
            metric_score: "0.87",
            distill_teacher_metric_name: "accuracy",
            distill_teacher_metric_value: "0.9",
            distill_student_metric_value: "0.87",
            distill_quality_delta: "-0.03",
            distill_higher_is_better: "1",
            distill_teacher_training_mode: "mlp_dense",
            distill_student_training_mode: "mlp_dense",
          },
        ],
        prependTrainingRun: vi.fn(),
        ui,
        runTraining: vi.fn(),
        runDistillation: vi.fn(),
      })
    );

    act(() => {
      result.current.onSeeDistilledFromRun({ run_id: "teacher-fallback", result: "completed" });
    });

    expect(ui.setDistillMetrics).toHaveBeenCalled();
    expect(ui.setDistillComparison).toHaveBeenCalled();
    expect(ui.setIsDistillMetricsModalOpen).toHaveBeenCalledWith(true);
  });

  it("handles unsupported/no-dataset/error distillation branches", async () => {
    const ui = createUi({ trainingMode: "unsupported_mode", isTraining: true });
    const runDistillation = vi.fn(async () => ({ status: "error", error: "distill failed" }));
    const { result } = renderHook(() =>
      usePytorchLogic({
        dataset: createDataset({ selectedDatasetId: null }),
        trainingRuns: [],
        prependTrainingRun: vi.fn(),
        ui,
        runTraining: vi.fn(),
        runDistillation,
      })
    );

    act(() => {
      result.current.onStopTrainingRuns();
    });
    expect(ui.setTrainingError).toHaveBeenCalledWith(
      "Stop requested. Current run will finish, then remaining runs will be skipped."
    );

    await act(async () => {
      await result.current.onDistillFromRun({
        result: "completed",
        run_id: "teacher-1",
        model_id: "teacher-model",
        training_mode: "unsupported_mode",
      });
    });
    expect(ui.setTrainingError).toHaveBeenCalledWith(
      "Distillation is not supported for 'unsupported_mode' yet."
    );

    const ui2 = createUi();
    const { result: result2 } = renderHook(() =>
      usePytorchLogic({
        dataset: createDataset(),
        trainingRuns: [],
        prependTrainingRun: vi.fn(),
        ui: ui2,
        runTraining: vi.fn(),
        runDistillation,
      })
    );
    await act(async () => {
      await result2.current.onDistillFromRun({
        result: "completed",
        run_id: "teacher-2",
        model_id: "teacher-model",
        model_path: "/tmp/m",
        training_mode: "mlp_dense",
      });
    });
    expect(runDistillation).toHaveBeenCalled();
    expect(ui2.setTrainingError).toHaveBeenCalledWith("distill failed");
  });

  it("exposes sweep/optimizer helpers and distillation support predicate", () => {
    const ui = createUi();
    const { result } = renderHook(() =>
      usePytorchLogic({
        dataset: createDataset(),
        trainingRuns: [{ result: "completed", metric_name: "accuracy", metric_score: 0.9, training_mode: "mlp_dense" }],
        prependTrainingRun: vi.fn(),
        ui,
        runTraining: vi.fn(),
        runDistillation: vi.fn(),
        runtime: {
          notifySuccess: vi.fn(),
          notifyError: vi.fn(),
          schedule: (callback: () => void) => callback(),
          writeClipboardText: vi.fn(async () => undefined),
        },
      })
    );

    act(() => {
      result.current.toggleRunSweep(true);
      result.current.reloadSweepValues();
      result.current.onFindOptimalParamsClick();
      result.current.onApplyOptimalParams();
    });

    expect(ui.setRunSweepEnabled).toHaveBeenCalled();
    expect(result.current.isDistillationSupportedForRun({ training_mode: "mlp_dense" })).toBe(true);
    expect(result.current.isDistillationSupportedForRun({ training_mode: "unknown" })).toBe(false);
  });

  it("uses default runtime for toasts and clipboard error fallback", async () => {
    vi.useFakeTimers();
    const ui = createUi();
    const { result } = renderHook(() =>
      usePytorchLogic({
        dataset: createDataset(),
        trainingRuns: [{ run_id: "r1", result: "completed", metric_name: "accuracy", metric_score: 0.9 }],
        prependTrainingRun: vi.fn(),
        ui,
        runTraining: vi.fn(async () => ({
          stopped: false,
          completed: 1,
          total: 1,
          completedTeacherRuns: [],
          failedRuns: 0,
          firstFailureMessage: null,
        })),
        runDistillation: vi.fn(),
      })
    );

    await act(async () => {
      await result.current.onTrainClick();
      await result.current.onCopyTrainingRuns();
    });

    expect(toast.success).toHaveBeenCalledWith("Training sequence completed.");
    expect(ui.setCopyRunsStatus).toHaveBeenCalledWith("Copy failed");

    act(() => {
      vi.runAllTimers();
    });
    vi.useRealTimers();
  });

  it("handles stopped training and fallback error toast copy", async () => {
    const ui = createUi({ excludeColumnsInput: "col_a,col_b", dateColumnsInput: "created_at" });
    const { result } = renderHook(() =>
      usePytorchLogic({
        dataset: createDataset(),
        trainingRuns: [],
        prependTrainingRun: vi.fn(),
        ui,
        runTraining: vi.fn(async (_problem, deps) => {
          deps.onProgress(1, 3);
          return {
            stopped: true,
            completed: 1,
            total: 3,
            completedTeacherRuns: [],
            failedRuns: 0,
            firstFailureMessage: null,
          };
        }),
        runDistillation: vi.fn(),
      })
    );

    await act(async () => {
      await result.current.onTrainClick();
    });

    expect(ui.setTrainingProgress).toHaveBeenCalledWith({ current: 1, total: 3 });
    expect(ui.setTrainingError).toHaveBeenCalledWith("Training stopped after 1/3 run(s).");
  });

  it("runs supported auto-distillation from training outcomes", async () => {
    const ui = createUi({ autoDistillEnabled: true, trainingMode: "mlp_dense" });
    const runDistillation = vi.fn(async () => ({
      status: "ok",
      metrics: { test_metric_name: "loss" },
      modelId: "student-auto-1",
      modelPath: "/tmp/student-auto-1",
      runId: "distill-auto-1",
      teacherModelSizeBytes: null,
      studentModelSizeBytes: null,
      teacherInputDim: null,
      teacherOutputDim: null,
      studentInputDim: null,
      studentOutputDim: null,
      sizeSavedBytes: null,
      sizeSavedPercent: null,
      teacherParamCount: null,
      studentParamCount: null,
      paramSavedCount: null,
      paramSavedPercent: null,
      distilledRun: {
        result: "distilled",
        run_id: "distill-auto-1",
        model_id: "student-auto-1",
        model_path: "/tmp/student-auto-1",
        metric_name: "loss",
        metric_score: "n/a",
        hidden_dim: "n/a",
        num_hidden_layers: "n/a",
      },
    }));

    const { result } = renderHook(() =>
      usePytorchLogic({
        dataset: createDataset(),
        trainingRuns: [],
        prependTrainingRun: vi.fn(),
        ui,
        runTraining: vi.fn(async () => ({
          stopped: false,
          completed: 1,
          total: 1,
          completedTeacherRuns: [
            {
              result: "completed",
              run_id: "teacher-auto-1",
              model_id: "teacher-model-1",
              model_path: "/tmp/teacher-model-1",
              metric_name: "loss",
              metric_score: 0.3,
              training_mode: "mlp_dense",
            },
          ],
          failedRuns: 1,
          firstFailureMessage: null,
        })),
        runDistillation,
      })
    );

    await act(async () => {
      await result.current.onTrainClick();
    });

    expect(toast.error).toHaveBeenCalledWith("1 training run(s) failed in the sequence.");
    expect(runDistillation).toHaveBeenCalledTimes(1);
  });

  it("surfaces missing teacher reference for manual distillation", async () => {
    const ui = createUi();
    const { result } = renderHook(() =>
      usePytorchLogic({
        dataset: createDataset(),
        trainingRuns: [],
        prependTrainingRun: vi.fn(),
        ui,
        runTraining: vi.fn(),
        runDistillation: vi.fn(),
      })
    );

    await act(async () => {
      await result.current.onDistillFromRun({
        result: "completed",
        run_id: "n/a",
        model_id: "n/a",
        model_path: "n/a",
        training_mode: "mlp_dense",
      });
    });

    expect(ui.setTrainingError).toHaveBeenCalledWith(
      "This run has no teacher model reference to distill from."
    );
  });

  it("uses default clipboard runtime success path", async () => {
    const writeText = vi.fn(async () => undefined);
    Object.defineProperty(globalThis.navigator, "clipboard", {
      value: { writeText },
      configurable: true,
    });

    const ui = createUi();
    const { result } = renderHook(() =>
      usePytorchLogic({
        dataset: createDataset(),
        trainingRuns: [{ run_id: "r1", result: "completed" }],
        prependTrainingRun: vi.fn(),
        ui,
        runTraining: vi.fn(),
        runDistillation: vi.fn(),
      })
    );

    await act(async () => {
      await result.current.onCopyTrainingRuns();
    });

    expect(writeText).toHaveBeenCalledTimes(1);
    expect(ui.setCopyRunsStatus).toHaveBeenCalledWith("Copied");
  });
});
