import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/ui/patterns/CsvDatasetCombobox", () => ({
  default: ({
    selectedId,
    emptyMessage,
    onChange,
  }: {
    selectedId: string | null;
    emptyMessage?: string;
    onChange: (value: string) => void;
  }) => (
    <div>
      <span data-testid="dataset-combobox">dataset:{selectedId ?? "none"}</span>
      {emptyMessage ? <span>{emptyMessage}</span> : null}
      <button type="button" onClick={() => onChange("next.csv")}>
        Change dataset
      </button>
    </div>
  ),
}));

vi.mock("@/ui/components/Datatable/DataTable", () => ({
  default: ({ rows, columns }: { rows: unknown[]; columns: string[] }) => (
    <div data-testid="data-table">
      rows:{rows.length} columns:{columns.join(",")}
    </div>
  ),
}));

vi.mock("@/features/ml/react/views/components/FieldHelp", () => ({
  FieldHelp: ({ text }: { text: string }) => <span title={text}>?</span>,
}));

vi.mock("@/features/ml/react/views/components/TrainingRunsSection", () => ({
  TrainingRunsSection: ({
    trainingRuns,
    copyRunsStatus,
    onCopyTrainingRuns,
    onClearTrainingRuns,
    onStopTrainingRuns,
  }: {
    trainingRuns: unknown[];
    copyRunsStatus: string | null;
    onCopyTrainingRuns: () => void;
    onClearTrainingRuns: () => void;
    onStopTrainingRuns: () => void;
  }) => (
    <section aria-label="training runs">
      <span>runs:{trainingRuns.length}</span>
      {copyRunsStatus ? <span>{copyRunsStatus}</span> : null}
      <button type="button" onClick={onCopyTrainingRuns}>
        Copy runs
      </button>
      <button type="button" onClick={onClearTrainingRuns}>
        Clear runs
      </button>
      <button type="button" onClick={onStopTrainingRuns}>
        Stop runs
      </button>
    </section>
  ),
}));

vi.mock("@/features/ml/react/views/components/MlTrainingModals", () => ({
  OptimalParamsModal: ({
    isOpen,
    onClose,
    onApply,
    activeAlgorithm,
  }: {
    isOpen: boolean;
    onClose: () => void;
    onApply: () => void;
    activeAlgorithm: string;
  }) =>
    isOpen ? (
      <div role="dialog" aria-label="optimal params">
        <span>algorithm:{activeAlgorithm}</span>
        <button type="button" onClick={onApply}>
          Apply optimal
        </button>
        <button type="button" onClick={onClose}>
          Close optimal
        </button>
      </div>
    ) : null,
  DistillMetricsModal: ({
    isOpen,
    onClose,
    distillModelId,
    distillModelPath,
  }: {
    isOpen: boolean;
    onClose: () => void;
    distillModelId: string | null;
    distillModelPath: string | null;
  }) =>
    isOpen ? (
      <div role="dialog" aria-label="distill metrics">
        <span>distill:{distillModelId ?? "none"}:{distillModelPath ?? "none"}</span>
        <button type="button" onClick={onClose}>
          Close distill
        </button>
      </div>
    ) : null,
}));

vi.mock("@/features/ml-model-ui/react/views/components/ModelPreviewModal", () => ({
  ModelPreviewModal: ({
    isOpen,
    onClose,
    framework,
    mode,
  }: {
    isOpen: boolean;
    onClose: () => void;
    framework: string;
    mode: string;
  }) =>
    isOpen ? (
      <div role="dialog" aria-label="model preview">
        <span>
          preview:{framework}:{mode}
        </span>
        <button type="button" onClick={onClose}>
          Close model
        </button>
      </div>
    ) : null,
}));

const pytorchBridgeMock = vi.fn();
const tensorflowBridgeMock = vi.fn();

vi.mock("@/features/ml/react/ai/tools/usePytorchFormBridge.tools", () => ({
  usePytorchFormBridge: (bindings: unknown) => pytorchBridgeMock(bindings),
}));

vi.mock("@/features/ml/react/ai/tools/useTensorflowFormBridge.tools", () => ({
  useTensorflowFormBridge: (bindings: unknown) => tensorflowBridgeMock(bindings),
}));

import PytorchTrainingScreen from "@/features/ml/react/views/screens/PytorchTrainingScreen";
import TensorflowTrainingScreen from "@/features/ml/react/views/screens/TensorflowTrainingScreen";

type TrainingScreenState = ReturnType<typeof createTrainingScreenState>;

function okValues(values: number[]) {
  return { ok: true as const, values };
}

function errorValue(error: string) {
  return { ok: false as const, error };
}

function createTrainingScreenState(overrides: Record<string, unknown> = {}) {
  return {
    datasetOptions: [{ id: "demo.csv", label: "Demo" }],
    selectedDatasetId: "demo.csv",
    isLoading: false,
    error: null,
    tableRows: [{ target: 1, feature: 2 }],
    tableColumns: ["target", "feature"],
    rowCount: 1,
    totalRowCount: 3,
    trainingMode: "mlp_dense",
    setTrainingMode: vi.fn(),
    isLinearBaselineMode: false,
    isStopRequested: false,
    targetColumn: "target",
    setTargetColumn: vi.fn(),
    resolvedExcludeColumnsInput: "id",
    setExcludeColumnsInput: vi.fn(),
    resolvedDateColumnsInput: "created_at",
    setDateColumnsInput: vi.fn(),
    task: "classification",
    setTask: vi.fn(),
    epochValuesInput: "10",
    setEpochValuesInput: vi.fn(),
    testSizesInput: "0.2",
    setTestSizesInput: vi.fn(),
    learningRatesInput: "0.001",
    setLearningRatesInput: vi.fn(),
    batchSizesInput: "32",
    setBatchSizesInput: vi.fn(),
    hiddenDimsInput: "128",
    setHiddenDimsInput: vi.fn(),
    numHiddenLayersInput: "2",
    setNumHiddenLayersInput: vi.fn(),
    dropoutsInput: "0.1",
    setDropoutsInput: vi.fn(),
    runSweepEnabled: true,
    toggleRunSweep: vi.fn(),
    reloadSweepValues: vi.fn(),
    isTraining: false,
    isDistilling: false,
    autoDistillEnabled: false,
    setAutoDistillEnabled: vi.fn(),
    distillingTeacherKey: null,
    distilledByTeacher: {},
    trainingProgress: { current: 0, total: 0 },
    trainingError: null,
    plannedRunCount: 1,
    epochsValidation: okValues([10]),
    testSizesValidation: okValues([0.2]),
    learningRatesValidation: okValues([0.001]),
    batchSizesValidation: okValues([32]),
    hiddenDimsValidation: okValues([128]),
    numHiddenLayersValidation: okValues([2]),
    dropoutsValidation: okValues([0.1]),
    defaults: { targetColumn: "target", excludeColumns: ["id"], dateColumns: ["created_at"] },
    onDatasetChange: vi.fn(),
    onTrainClick: vi.fn(),
    onFindOptimalParamsClick: vi.fn(),
    onApplyOptimalParams: vi.fn(),
    onStopTrainingRuns: vi.fn(),
    onDistillFromRun: vi.fn(),
    onSeeDistilledFromRun: vi.fn(),
    isDistillationSupportedForRun: vi.fn(() => true),
    trainingRuns: [{ run_id: "r1" }],
    copyRunsStatus: "Copied",
    onCopyTrainingRuns: vi.fn(),
    clearTrainingRuns: vi.fn(),
    completedRuns: [{ run_id: "r1" }, { run_id: "r2" }, { run_id: "r3" }, { run_id: "r4" }, { run_id: "r5" }],
    optimizerStatus: "ready",
    distillStatus: "distill ready",
    isOptimalModalOpen: true,
    setIsOptimalModalOpen: vi.fn(),
    pendingOptimalParams: { epochs: 10 },
    pendingOptimalPrediction: { metricName: "accuracy", metricValue: 0.9 },
    isDistillMetricsModalOpen: true,
    setIsDistillMetricsModalOpen: vi.fn(),
    distillMetrics: { test_metric_name: "accuracy", test_metric_value: 0.9 },
    distillModelId: "student-1",
    distillModelPath: "/tmp/student",
    distillComparison: null,
    ...overrides,
  };
}

async function exerciseTrainingScreen(state: TrainingScreenState, prefix: "pytorch" | "tensorflow") {
  const user = userEvent.setup();
  const setInputValue = async (selector: string, value: string) => {
    const input = document.querySelector(selector) as HTMLInputElement;
    await user.clear(input);
    await user.type(input, value);
  };

  await user.click(screen.getAllByRole("button", { name: "Change dataset" })[0]);
  await user.selectOptions(
    document.querySelector(`[data-ai-field="${prefix}_training_mode"]`) as HTMLSelectElement,
    prefix === "pytorch" ? "tabresnet" : "entity_embeddings"
  );
  await user.selectOptions(
    document.querySelector(`[data-ai-field="${prefix}_target_column"]`) as HTMLSelectElement,
    "feature"
  );
  await user.selectOptions(screen.getAllByRole("combobox")[2], "regression");
  await setInputValue(`[data-ai-field="${prefix}_epoch_values"], input[placeholder="e.g. 10,20,50,100,200"]`, "20");
  await setInputValue(`[data-ai-field="${prefix}_batch_sizes"], input[placeholder="e.g. 32,64,128"]`, "16");
  await setInputValue(`[data-ai-field="${prefix}_learning_rates"], input[placeholder="e.g. 0.001,0.0005"]`, "0.01");
  await setInputValue(`[data-ai-field="${prefix}_test_sizes"], input[placeholder="e.g. 0.2,0.3"]`, "0.25");
  await setInputValue(`[data-ai-field="${prefix}_hidden_dims"], input[placeholder="e.g. 128,256"]`, "64");
  await setInputValue(`[data-ai-field="${prefix}_num_hidden_layers"], input[placeholder="e.g. 2,3,4"]`, "3");
  await setInputValue(`[data-ai-field="${prefix}_dropouts"], input[placeholder="e.g. 0.1,0.2"]`, "0.2");
  await setInputValue(`[data-ai-field="${prefix}_exclude_columns"], input[placeholder="e.g. customerID,Order,PID"]`, "uuid");
  await setInputValue(`[data-ai-field="${prefix}_date_columns"], input[placeholder="e.g. Date"]`, "created");
  await user.click(screen.getByRole("button", { name: "Train Model" }));
  await user.click(screen.getByRole("button", { name: "Find Optimal Params" }));
  await user.click(screen.getByRole("checkbox", { name: /Set Sweep Values/i }));
  await user.click(screen.getByRole("button", { name: "Reload" }));
  await user.click(screen.getByRole("checkbox", { name: /Auto-distill Training Runs/i }));
  await user.click(screen.getByRole("button", { name: "Copy runs" }));
  await user.click(screen.getByRole("button", { name: "Clear runs" }));
  await user.click(screen.getByRole("button", { name: "Stop runs" }));
  await user.click(screen.getByRole("button", { name: "Apply optimal" }));
  await user.click(screen.getByRole("button", { name: "Close optimal" }));
  await user.click(screen.getByRole("button", { name: "Close distill" }));
  await user.click(screen.getByRole("button", { name: "Show Model" }));
  await user.click(screen.getByRole("button", { name: "Close model" }));

  expect(state.onDatasetChange).toHaveBeenCalledWith("next.csv");
  expect(state.setTrainingMode).toHaveBeenCalledWith(
    prefix === "pytorch" ? "tabresnet" : "entity_embeddings"
  );
  expect(state.setTargetColumn).toHaveBeenCalledWith("feature");
  expect(state.setTask).toHaveBeenCalledWith("regression");
  expect(state.setEpochValuesInput).toHaveBeenCalled();
  expect(state.setBatchSizesInput).toHaveBeenCalled();
  expect(state.setLearningRatesInput).toHaveBeenCalled();
  expect(state.setTestSizesInput).toHaveBeenCalled();
  expect(state.setHiddenDimsInput).toHaveBeenCalled();
  expect(state.setNumHiddenLayersInput).toHaveBeenCalled();
  expect(state.setDropoutsInput).toHaveBeenCalled();
  expect(state.setExcludeColumnsInput).toHaveBeenCalled();
  expect(state.setDateColumnsInput).toHaveBeenCalled();
  expect(state.onTrainClick).toHaveBeenCalledTimes(1);
  expect(state.onFindOptimalParamsClick).toHaveBeenCalledTimes(1);
  expect(state.toggleRunSweep).toHaveBeenCalledWith(false);
  expect(state.reloadSweepValues).toHaveBeenCalledTimes(1);
  expect(state.setAutoDistillEnabled).toHaveBeenCalledWith(true);
  expect(state.onCopyTrainingRuns).toHaveBeenCalledTimes(1);
  expect(state.clearTrainingRuns).toHaveBeenCalledTimes(1);
  expect(state.onStopTrainingRuns).toHaveBeenCalledTimes(1);
  expect(state.onApplyOptimalParams).toHaveBeenCalledTimes(1);
}

describe("ML training screens", () => {
  afterEach(() => cleanup());

  beforeEach(() => {
    pytorchBridgeMock.mockClear();
    tensorflowBridgeMock.mockClear();
  });

  it("wires PyTorch screen controls, modals, table preview, and bridge bindings", async () => {
    const state = createTrainingScreenState();
    const orchestrator = vi.fn(() => state);

    render(<PytorchTrainingScreen orchestrator={orchestrator} />);

    expect(screen.getByText("Machine Learning with PyTorch")).toBeInTheDocument();
    expect(screen.getByText(/Showing 1 rows/)).toBeInTheDocument();
    expect(screen.getByTestId("data-table")).toHaveTextContent("columns:target,feature");
    expect(screen.getByText("algorithm:mlp_dense")).toBeInTheDocument();
    expect(screen.getByText("distill:student-1:/tmp/student")).toBeInTheDocument();
    expect(pytorchBridgeMock).toHaveBeenCalledWith(
      expect.objectContaining({
        trainingMode: "mlp_dense",
        setDatasetId: expect.any(Function),
        onTrainClick: state.onTrainClick,
      })
    );

    await exerciseTrainingScreen(state, "pytorch");
  });

  it("renders PyTorch loading/error and linear-baseline disabled states", () => {
    const state = createTrainingScreenState({
      selectedDatasetId: null,
      isLoading: true,
      error: "Backend unavailable",
      isLinearBaselineMode: true,
      isTraining: true,
      trainingProgress: { current: 1, total: 2 },
      plannedRunCount: 0,
      defaults: { targetColumn: "", excludeColumns: [], dateColumns: [] },
      epochsValidation: errorValue("bad epochs"),
      hiddenDimsValidation: errorValue("bad hidden"),
      optimizerStatus: null,
      distillStatus: null,
      trainingError: "Training failed",
      isOptimalModalOpen: false,
      isDistillMetricsModalOpen: false,
    });

    render(<PytorchTrainingScreen orchestrator={vi.fn(() => state)} />);

    expect(screen.getAllByText("Backend unavailable")).toHaveLength(2);
    expect(screen.getByRole("button", { name: "Training 1/2..." })).toBeDisabled();
    expect(screen.getByText("Epochs: bad epochs")).toHaveClass("text-red-600");
    expect(screen.getByText("Hidden dims: n/a (linear baseline)")).toBeInTheDocument();
    expect(screen.getAllByText("Preloaded: (none)")).toHaveLength(2);
    expect(screen.getByText("Training failed")).toBeInTheDocument();
  });

  it("renders PyTorch fallback copy and non-linear validation errors", () => {
    const state = createTrainingScreenState({
      selectedDatasetId: null,
      isLoading: true,
      error: null,
      rowCount: 3,
      totalRowCount: 3,
      plannedRunCount: 0,
      defaults: { targetColumn: "", excludeColumns: [], dateColumns: [] },
      batchSizesValidation: errorValue("bad batch sizes"),
      learningRatesValidation: errorValue("bad learning rates"),
      testSizesValidation: errorValue("bad test sizes"),
      hiddenDimsValidation: errorValue("bad hidden dims"),
      numHiddenLayersValidation: errorValue("bad layers"),
      dropoutsValidation: errorValue("bad dropouts"),
      isOptimalModalOpen: false,
      isDistillMetricsModalOpen: false,
    });

    render(<PytorchTrainingScreen orchestrator={vi.fn(() => state)} />);

    expect(screen.getByText("Loading datasets...")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Train Model" })).toBeDisabled();
    expect(screen.getByText("Batch sizes: bad batch sizes")).toHaveClass("text-red-600");
    expect(screen.getByText("Learning rates: bad learning rates")).toHaveClass("text-red-600");
    expect(screen.getByText("Test sizes: bad test sizes")).toHaveClass("text-red-600");
    expect(screen.getByText("Hidden dims: bad hidden dims")).toHaveClass("text-red-600");
    expect(screen.getByText("Hidden layers: bad layers")).toHaveClass("text-red-600");
    expect(screen.getByText("Dropouts: bad dropouts")).toHaveClass("text-red-600");
    expect(screen.getByText(/Showing 3 rows for/)).toBeInTheDocument();
  });

  it("wires TensorFlow screen controls, modals, table preview, and bridge bindings", async () => {
    const state = createTrainingScreenState({ trainingMode: "wide_and_deep" });
    const orchestrator = vi.fn(() => state);

    render(<TensorflowTrainingScreen orchestrator={orchestrator} />);

    expect(screen.getByText("Machine Learning with TensorFlow")).toBeInTheDocument();
    expect(screen.getByText("algorithm:wide_and_deep")).toBeInTheDocument();
    expect(tensorflowBridgeMock).toHaveBeenCalledWith(
      expect.objectContaining({
        trainingMode: "wide_and_deep",
        setDatasetId: expect.any(Function),
        onTrainClick: state.onTrainClick,
      })
    );

    await exerciseTrainingScreen(state, "tensorflow");
  });

  it("renders TensorFlow loading/error and linear-baseline disabled states", () => {
    const state = createTrainingScreenState({
      selectedDatasetId: null,
      isLoading: false,
      error: null,
      isLinearBaselineMode: true,
      isDistilling: true,
      trainingProgress: { current: 0, total: 0 },
      plannedRunCount: 0,
      defaults: { targetColumn: "", excludeColumns: [], dateColumns: [] },
      batchSizesValidation: errorValue("bad batch sizes"),
      learningRatesValidation: errorValue("bad learning rates"),
      testSizesValidation: errorValue("bad test sizes"),
      numHiddenLayersValidation: errorValue("bad layers"),
      dropoutsValidation: errorValue("bad dropouts"),
      optimizerStatus: null,
      distillStatus: null,
      trainingError: "Training failed",
      isOptimalModalOpen: false,
      isDistillMetricsModalOpen: false,
    });

    render(<TensorflowTrainingScreen orchestrator={vi.fn(() => state)} />);

    expect(screen.getByText("No dataset found.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Train Model" })).toBeDisabled();
    expect(screen.getByText("Batch sizes: bad batch sizes")).toHaveClass("text-red-600");
    expect(screen.getByText("Learning rates: bad learning rates")).toHaveClass("text-red-600");
    expect(screen.getByText("Test sizes: bad test sizes")).toHaveClass("text-red-600");
    expect(screen.getByText("Hidden layers: n/a (linear baseline)")).toBeInTheDocument();
    expect(screen.getByText("Dropouts: n/a (linear baseline)")).toBeInTheDocument();
    expect(screen.getByText("Training failed")).toBeInTheDocument();
  });

  it("renders TensorFlow training progress and backend error state", () => {
    const state = createTrainingScreenState({
      error: "Backend unavailable",
      isLoading: true,
      isTraining: true,
      trainingProgress: { current: 2, total: 4 },
      isOptimalModalOpen: false,
      isDistillMetricsModalOpen: false,
    });

    render(<TensorflowTrainingScreen orchestrator={vi.fn(() => state)} />);

    expect(screen.getAllByText("Backend unavailable")).toHaveLength(2);
    expect(screen.getByRole("button", { name: "Training 2/4..." })).toBeDisabled();
  });

  it("renders TensorFlow non-linear validation errors and exact preview row count", () => {
    const state = createTrainingScreenState({
      rowCount: 3,
      totalRowCount: 3,
      epochsValidation: errorValue("bad epochs"),
      hiddenDimsValidation: errorValue("bad hidden dims"),
      numHiddenLayersValidation: errorValue("bad layers"),
      dropoutsValidation: errorValue("bad dropouts"),
      isOptimalModalOpen: false,
      isDistillMetricsModalOpen: false,
    });

    render(<TensorflowTrainingScreen orchestrator={vi.fn(() => state)} />);

    expect(screen.getByText("Epochs: bad epochs")).toHaveClass("text-red-600");
    expect(screen.getByText("Hidden dims: bad hidden dims")).toHaveClass("text-red-600");
    expect(screen.getByText("Hidden layers: bad layers")).toHaveClass("text-red-600");
    expect(screen.getByText("Dropouts: bad dropouts")).toHaveClass("text-red-600");
    expect(screen.getByText(/Showing 3 rows for/)).toBeInTheDocument();
  });
});
