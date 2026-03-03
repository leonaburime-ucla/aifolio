import { act, renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { toast } from "react-hot-toast";
import {
  useTensorflowLogic,
  useTensorflowUiState,
} from "@/features/ml/typescript/react/hooks/useTensorflowTraining.hooks";
import type { TensorflowLogicArgs } from "@/features/ml/__types__/typescript/react/hooks/tensorflowTraining.types";

vi.mock("react-hot-toast", () => ({
  toast: {
    success: vi.fn(),
    error: vi.fn(),
  },
}));

function createUi(
  overrides: Record<string, unknown> = {}
): TensorflowLogicArgs["ui"] {
  return {
    trainingMode: "wide_and_deep",
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
  } as unknown as TensorflowLogicArgs["ui"];
}

function createDataset(
  overrides: Record<string, unknown> = {}
): TensorflowLogicArgs["dataset"] {
  return {
    selectedDatasetId: "customer_churn_telco.csv",
    setSelectedDatasetId: vi.fn(),
    tableColumns: ["Churn", "Age"],
    ...overrides,
  } as unknown as TensorflowLogicArgs["dataset"];
}

describe("useTensorflowTraining.hooks", () => {
  it("provides default ui mode", () => {
    const { result } = renderHook(() => useTensorflowUiState());
    expect(result.current.trainingMode).toBe("wide_and_deep");

    act(() => {
      result.current.setTrainingMode("mlp_dense");
    });
    expect(result.current.trainingMode).toBe("mlp_dense");
  });

  it("runs training and distillation success paths", async () => {
    const ui = createUi({ autoDistillEnabled: true });
    const dataset = createDataset();
    const runTraining = vi.fn(async () => ({
      stopped: false,
      completed: 1,
      total: 1,
      completedTeacherRuns: [
        {
          result: "completed",
          run_id: "teacher-1",
          model_id: "teacher-model",
          model_path: "/tmp/teacher",
          metric_name: "accuracy",
          metric_score: 0.85,
          hidden_dim: 128,
          num_hidden_layers: 3,
          dropout: 0.1,
          epochs: 60,
          batch_size: 64,
          learning_rate: 0.001,
          test_size: 0.2,
          training_mode: "wide_and_deep",
        },
      ],
      failedRuns: 0,
      firstFailureMessage: null,
    }));
    const prependTrainingRun = vi.fn();
    const runDistillation = vi.fn(async () => ({
      status: "ok",
      metrics: { test_metric_name: "accuracy", test_metric_value: 0.9 },
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
        metric_score: 0.9,
        hidden_dim: 32,
        num_hidden_layers: 2,
        training_mode: "wide_and_deep",
      },
    }));

    const { result } = renderHook(() =>
      useTensorflowLogic({
        dataset,
        trainingRuns: [],
        prependTrainingRun,
        ui,
        runTraining,
        runDistillation,
        runtime: {
          notifySuccess: vi.fn(),
          notifyError: vi.fn(),
          schedule: (callback: () => void) => callback(),
          writeClipboardText: vi.fn(async () => undefined),
        },
      })
    );

    await act(async () => {
      await result.current.onTrainClick();
    });

    act(() => {
      result.current.onSeeDistilledFromRun({ run_id: "teacher-1", result: "completed" });
    });

    expect(runTraining).toHaveBeenCalledTimes(1);
    expect(runDistillation).toHaveBeenCalledTimes(1);
    expect(prependTrainingRun).toHaveBeenCalledTimes(1);
    expect(ui.setDistillComparison).toHaveBeenCalled();
  });

  it("handles missing teacher model and fallback errors", async () => {
    const ui = createUi();
    const dataset = createDataset();

    const { result } = renderHook(() =>
      useTensorflowLogic({
        dataset,
        trainingRuns: [],
        prependTrainingRun: vi.fn(),
        ui,
        runTraining: vi.fn(),
        runDistillation: vi.fn(),
      })
    );

    await act(async () => {
      await result.current.onDistillFromRun({ result: "completed", run_id: "n/a" });
    });
    expect(ui.setTrainingError).toHaveBeenCalledWith("This run has no teacher model reference to distill from.");

    act(() => {
      result.current.onSeeDistilledFromRun({ run_id: "unknown", result: "completed" });
    });
    expect(ui.setTrainingError).toHaveBeenCalledWith("No distilled result found yet for this teacher run.");
  });

  it("handles unsupported/no-dataset/error distillation branches", async () => {
    const ui = createUi({ trainingMode: "unsupported_mode", isTraining: true });
    const runDistillation = vi.fn(async () => ({ status: "error", error: "tf distill failed" }));

    const { result } = renderHook(() =>
      useTensorflowLogic({
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
      useTensorflowLogic({
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
        training_mode: "wide_and_deep",
      });
    });
    expect(runDistillation).toHaveBeenCalled();
    expect(ui2.setTrainingError).toHaveBeenCalledWith("tf distill failed");
  });

  it("exposes sweep/optimizer helpers and distillation support predicate", () => {
    const ui = createUi();
    const { result } = renderHook(() =>
      useTensorflowLogic({
        dataset: createDataset(),
        trainingRuns: [{ result: "completed", metric_name: "accuracy", metric_score: 0.9, training_mode: "wide_and_deep" }],
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
    expect(result.current.isDistillationSupportedForRun({ training_mode: "wide_and_deep" })).toBe(true);
    expect(result.current.isDistillationSupportedForRun({ training_mode: "unknown" })).toBe(false);
  });

  it("resets values on dataset change", () => {
    const ui = createUi();
    const dataset = createDataset();
    const { result } = renderHook(() =>
      useTensorflowLogic({
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

  it("uses fallback distilled rows and default runtime clipboard branch", async () => {
    vi.useFakeTimers();
    const ui = createUi();
    const { result } = renderHook(() =>
      useTensorflowLogic({
        dataset: createDataset(),
        trainingRuns: [
          {
            result: "completed",
            run_id: "r1",
            metric_name: "accuracy",
            metric_score: 0.9,
          },
          {
            result: "distilled",
            teacher_ref_key: "teacher-fallback",
            run_id: "distilled-1",
            model_id: "model-distilled-1",
            model_path: "/tmp/distilled-1",
            metric_name: "accuracy",
            metric_score: "0.86",
            distill_teacher_metric_name: "accuracy",
            distill_teacher_metric_value: "0.9",
            distill_student_metric_value: "0.86",
            distill_quality_delta: "-0.04",
            distill_higher_is_better: "1",
            distill_teacher_training_mode: "wide_and_deep",
            distill_student_training_mode: "wide_and_deep",
          },
        ],
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

    act(() => {
      result.current.onSeeDistilledFromRun({ run_id: "teacher-fallback", result: "completed" });
    });
    expect(ui.setDistillComparison).toHaveBeenCalled();

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

  it("handles invalid setup and stopped training paths", async () => {
    const ui = createUi({ excludeColumnsInput: "id", dateColumnsInput: "created_at" });
    const { result } = renderHook(() =>
      useTensorflowLogic({
        dataset: createDataset({ selectedDatasetId: null, tableColumns: [] }),
        trainingRuns: [],
        prependTrainingRun: vi.fn(),
        ui,
        runTraining: vi.fn(async (_problem, deps) => {
          deps.onProgress(1, 2);
          return {
            stopped: true,
            completed: 1,
            total: 2,
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
    expect(ui.setTrainingError).toHaveBeenCalledTimes(1);

    const { result: result2 } = renderHook(() =>
      useTensorflowLogic({
        dataset: createDataset(),
        trainingRuns: [],
        prependTrainingRun: vi.fn(),
        ui: createUi(),
        runTraining: vi.fn(async () => ({
          stopped: true,
          completed: 1,
          total: 2,
          completedTeacherRuns: [],
          failedRuns: 0,
          firstFailureMessage: null,
        })),
        runDistillation: vi.fn(),
      })
    );
    await act(async () => {
      await result2.current.onTrainClick();
    });
    expect(result2.current.isStopRequested).toBe(false);
  });

  it("runs supported auto-distillation and fallback notify error copy", async () => {
    const ui = createUi({ autoDistillEnabled: true, trainingMode: "wide_and_deep" });
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
      useTensorflowLogic({
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
              training_mode: "wide_and_deep",
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

  it("handles partial-success notify and unsupported auto-distill skip", async () => {
    const ui = createUi({ autoDistillEnabled: true, trainingMode: "unsupported_mode" });
    const runtime = {
      notifySuccess: vi.fn(),
      notifyError: vi.fn(),
      schedule: vi.fn((callback: () => void) => callback()),
      writeClipboardText: vi.fn(async () => undefined),
    };

    const { result } = renderHook(() =>
      useTensorflowLogic({
        dataset: createDataset(),
        trainingRuns: [],
        prependTrainingRun: vi.fn(),
        ui,
        runTraining: vi.fn(async (_problem, deps) => {
          deps.onProgress(1, 2);
          return {
            stopped: false,
            completed: 2,
            total: 2,
            completedTeacherRuns: [{ run_id: "teacher-x", result: "completed" }],
            failedRuns: 1,
            firstFailureMessage: null,
          };
        }),
        runDistillation: vi.fn(),
        runtime,
      })
    );

    await act(async () => {
      await result.current.onTrainClick();
    });

    expect(runtime.notifyError).toHaveBeenCalledWith(
      "1 training run(s) failed in the sequence."
    );
    expect(runtime.notifySuccess).toHaveBeenCalledWith(
      "Training sequence completed with partial success (1/2)."
    );
    expect(ui.setDistillStatus).toHaveBeenCalledWith(
      "Auto-distill skipped: 'unsupported_mode' distillation is not supported yet."
    );
  });
});
