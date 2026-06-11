import type { MlTaskType } from "../config/types";

// --- Domain types (mlData) ---

export type MlDatasetOption = {
  id: string;
  label: string;
  description?: string;
};

export type MlDatasetCacheEntry = {
  columns: string[];
  rows: Array<Record<string, string | number | null>>;
  rowCount: number;
  totalRowCount: number;
};

export type MlDatasetState = {
  datasetOptions: MlDatasetOption[];
  selectedDatasetId: string | null;
  datasetCache: Record<string, MlDatasetCacheEntry>;
  manifestLoaded: boolean;
  isLoadingManifest: boolean;
  isLoadingDataset: boolean;
  error: string | null;
};

export type MlDatasetActions = {
  setDatasetOptions: (value: MlDatasetOption[]) => void;
  setSelectedDatasetId: (value: string | null) => void;
  setDatasetCacheEntry: (datasetId: string, value: MlDatasetCacheEntry) => void;
  setManifestLoaded: (value: boolean) => void;
  setLoadingManifest: (value: boolean) => void;
  setLoadingDataset: (value: boolean) => void;
  setError: (value: string | null) => void;
};

// --- AG-UI tooling types ---

export type MlTask = "auto" | "classification" | "regression";
export type MlListField = string | number | number[];
export type MlMode = string;
export type MlFrameworkTab = "pytorch" | "tensorflow";

export type PytorchTrainingMode =
  | "mlp_dense"
  | "linear_glm_baseline"
  | "tabresnet"
  | "imbalance_aware"
  | "calibrated_classifier"
  | "tree_teacher_distillation";

export type TensorflowTrainingMode =
  | "mlp_dense"
  | "linear_glm_baseline"
  | "wide_and_deep"
  | "imbalance_aware"
  | "quantile_regression"
  | "calibrated_classifier"
  | "entity_embeddings"
  | "autoencoder_head"
  | "multi_task_learning"
  | "time_aware_tabular";

export type MlFormPatch<TMode extends MlMode = MlMode> = {
  dataset_id?: string;
  training_mode?: TMode;
  target_column?: string;
  task?: MlTask;
  epoch_values?: MlListField;
  batch_sizes?: MlListField;
  learning_rates?: MlListField;
  test_sizes?: MlListField;
  hidden_dims?: MlListField;
  num_hidden_layers?: MlListField;
  dropouts?: MlListField;
  exclude_columns?: string | string[];
  date_columns?: string | string[];
  set_sweep_values?: boolean;
  run_sweep?: boolean;
  auto_distill?: boolean;
};

export type PytorchFormPatch = MlFormPatch<PytorchTrainingMode>;
export type TensorflowFormPatch = MlFormPatch<TensorflowTrainingMode>;

export type MlFormRandomizeArgs = {
  confirm_randomize?: boolean;
  value_count?: number;
  style?: "safe" | "balanced" | "aggressive";
  set_sweep_values?: boolean;
  run_sweep?: boolean;
  auto_distill?: boolean;
  lock_target_column?: boolean;
  randomize_model_fields?: boolean;
};

export type PytorchRandomizeArgs = MlFormRandomizeArgs;
export type TensorflowRandomizeArgs = MlFormRandomizeArgs;

export type MlFormPatchResult = { applied: string[]; skipped: string[] };
export type MlFormStartResult =
  | { status: "ok" }
  | { status: "error"; reason: string };

export type MlFormBridge<TPatch extends MlFormPatch = MlFormPatch> = {
  applyPatch: (patch: TPatch) => MlFormPatchResult;
  startTrainingRuns?: () => Promise<MlFormStartResult>;
};

export type PytorchFormBridge = MlFormBridge<PytorchFormPatch>;
export type TensorflowFormBridge = MlFormBridge<TensorflowFormPatch>;

export type PytorchBridgePatch = PytorchFormPatch;
export type TensorflowBridgePatch = TensorflowFormPatch;
export type PytorchBridgeApplyResult = MlFormPatchResult;
export type TensorflowBridgeApplyResult = MlFormPatchResult;
export type PytorchBridgeStartResult = MlFormStartResult;
export type TensorflowBridgeStartResult = MlFormStartResult;

export type MlFormPatchBindings<TMode extends MlMode = MlMode> = {
  setDatasetId: (value: string) => void;
  setTrainingMode: (value: TMode) => void;
  setTargetColumn: (value: string) => void;
  setTask: (value: MlTaskType) => void;
  runSweepEnabled: boolean;
  toggleRunSweep: (enabled: boolean) => void;
  setEpochValuesInput: (value: string) => void;
  setBatchSizesInput: (value: string) => void;
  setLearningRatesInput: (value: string) => void;
  setTestSizesInput: (value: string) => void;
  setHiddenDimsInput: (value: string) => void;
  setNumHiddenLayersInput: (value: string) => void;
  setDropoutsInput: (value: string) => void;
  setExcludeColumnsInput: (value: string) => void;
  setDateColumnsInput: (value: string) => void;
  autoDistillEnabled: boolean;
  setAutoDistillEnabled: (enabled: boolean) => void;
};

export type PytorchBridgePatchBindings =
  MlFormPatchBindings<PytorchTrainingMode>;
export type TensorflowBridgePatchBindings =
  MlFormPatchBindings<TensorflowTrainingMode>;

export type MlFormBridgeBindings<
  TMode extends MlMode = MlMode,
> = MlFormPatchBindings<TMode> & {
  trainingMode: TMode;
  onTrainClick: () => Promise<void>;
};

export type PytorchFormBridgeBindings =
  MlFormBridgeBindings<PytorchTrainingMode>;
export type TensorflowFormBridgeBindings =
  MlFormBridgeBindings<TensorflowTrainingMode>;

export type TrainPytorchModelArgs = {
  dataset_id: string;
  target_column: string;
  task?: MlTask;
  epochs?: number;
  batch_size?: number;
  learning_rate?: number;
};

export type TrainTensorflowModelArgs = {
  dataset_id: string;
  target_column: string;
  task?: MlTask;
  epochs?: number;
  batch_size?: number;
  learning_rate?: number;
};

export type MlTargetColumnChangeMode = "different" | "random" | "next";

export type MlTargetColumnChangeArgs = {
  target_column?: string;
  mode?: MlTargetColumnChangeMode;
};

export type ChangePytorchTargetColumnArgs = MlTargetColumnChangeArgs;
export type ChangeTensorflowTargetColumnArgs = MlTargetColumnChangeArgs;

// --- Validator types ---

export type ValidationResult<T> =
  | { ok: true; values: T[] }
  | { ok: false; error: string };

export type SweepConfig = {
  epochs: number[];
  testSizes: number[];
  learningRates: number[];
  batchSizes: number[];
  hiddenDims: number[];
  numHiddenLayers: number[];
  dropouts: number[];
};

export type SweepCombination = {
  epochs: number;
  testSize: number;
  learningRate: number;
  batchSize: number;
  hiddenDim: number;
  numHiddenLayers: number;
  dropout: number;
};

// --- Utils types ---

export type HyperParams = {
  epochs: number;
  learning_rate: number;
  test_size: number;
  batch_size: number;
  hidden_dim: number;
  num_hidden_layers: number;
  dropout: number;
};

export type ParsedRun = HyperParams & {
  metric_name: string;
  metric_score: number;
};

export type ParamKey = keyof HyperParams;

export type ParamSpec = {
  key: ParamKey;
  type: "int" | "float";
  min: number;
  max: number;
};

export type OptimalParamsSuggestion = {
  suggestion: HyperParams;
  basedOnRuns: number;
  predictedMetricName: string;
  predictedMetricValue: number;
};

export type BayesianOptimizerRuntime = {
  random: () => number;
};

export type TrainingRunRow = Record<string, string | number | null>;

export type TrainingMetrics = {
  task?: string;
  train_loss?: number;
  test_loss?: number;
  test_metric_name?: string;
  test_metric_value?: number;
};

export type DistillComparison = {
  metricName: string;
  teacherMetricValue: number | null;
  studentMetricValue: number | null;
  qualityDelta: number | null;
  higherIsBetter: boolean;
  teacherTrainingMode: string | null;
  studentTrainingMode: string | null;
  teacherHiddenDim: number | null;
  studentHiddenDim: number | null;
  teacherNumHiddenLayers: number | null;
  studentNumHiddenLayers: number | null;
  teacherInputDim: number | null;
  studentInputDim: number | null;
  teacherOutputDim: number | null;
  studentOutputDim: number | null;
  teacherModelSizeBytes: number | null;
  studentModelSizeBytes: number | null;
  sizeSavedBytes: number | null;
  sizeSavedPercent: number | null;
  teacherParamCount: number | null;
  studentParamCount: number | null;
  paramSavedCount: number | null;
  paramSavedPercent: number | null;
};

export type TrainingProgress = {
  current: number;
  total: number;
};

export type NumericInputSnapshot = {
  epochValuesInput: string;
  batchSizesInput: string;
  learningRatesInput: string;
  testSizesInput: string;
  hiddenDimsInput: string;
  numHiddenLayersInput: string;
  dropoutsInput: string;
};

export type NumericInputSetters = {
  setEpochValuesInput: (value: string) => void;
  setBatchSizesInput: (value: string) => void;
  setLearningRatesInput: (value: string) => void;
  setTestSizesInput: (value: string) => void;
  setHiddenDimsInput: (value: string) => void;
  setNumHiddenLayersInput: (value: string) => void;
  setDropoutsInput: (value: string) => void;
};

export type RandomValueProvider = () => number;

export type DistilledSnapshot = {
  metrics: TrainingMetrics;
  modelId: string | null;
  modelPath: string | null;
};

export type DistilledSnapshotByTeacher = Record<
  string,
  {
    metrics: TrainingMetrics;
    modelId: string | null;
    modelPath: string | null;
    comparison: DistillComparison;
  }
>;

export type MlDatasetStatePort = {
  state: MlDatasetState;
  actions: MlDatasetActions;
};

export type MlActiveDataset = MlDatasetCacheEntry | null;

export type MlDatasetViewModel = {
  datasetOptions: { id: string; label: string }[];
  selectedDatasetId: string | null;
  setSelectedDatasetId: (datasetId: string | null) => void;
  isLoading: boolean;
  error: string | null;
  tableRows: Array<Record<string, string | number | null>>;
  tableColumns: string[];
  rowCount: number;
  totalRowCount: number;
  reloadManifest: () => Promise<void>;
  reloadDataset: () => Promise<void>;
};

export type TrainingRunsState = {
  trainingRuns: TrainingRunRow[];
  prependTrainingRun: (row: TrainingRunRow) => void;
  clearTrainingRuns: () => void;
};

export type UseTrainingRunsState = () => TrainingRunsState;

export type IntegrationComposeArgs<
  TDatasetState extends object,
  TUiState extends object,
  TLogic extends object,
> = {
  useDatasetState: () => TDatasetState;
  useUiState: () => TUiState;
  useLogic: (args: {
    dataset: TDatasetState;
    trainingRuns: TrainingRunRow[];
    prependTrainingRun: (row: TrainingRunRow) => void;
    ui: TUiState;
  }) => TLogic;
  useTrainingRunsState: UseTrainingRunsState;
};

export type MlTrainingRunsState = {
  trainingRuns: TrainingRunRow[];
};

export type MlTrainingRunsActions = {
  setTrainingRuns: (runs: TrainingRunRow[]) => void;
  prependTrainingRun: (run: TrainingRunRow) => void;
  clearTrainingRuns: () => void;
};

export type MlTrainingRunsStore =
  MlTrainingRunsState & MlTrainingRunsActions;

// --- Logic types ---

export type BaseTrainingRuntimeDeps = {
  notifySuccess: (message: string) => void;
  notifyError: (message: string) => void;
  schedule: (callback: () => void, delayMs: number) => void;
  writeClipboardText: (text: string) => Promise<void>;
};

export type TrainingInputValidations = {
  epochsValidation: ValidationResult<number>;
  testSizesValidation: ValidationResult<number>;
  learningRatesValidation: ValidationResult<number>;
  batchSizesValidation: ValidationResult<number>;
  hiddenDimsValidation: ValidationResult<number>;
  numHiddenLayersValidation: ValidationResult<number>;
  dropoutsValidation: ValidationResult<number>;
};

export type ResolveTargetColumnParams = {
  targetColumn: string;
  defaultTargetColumn: string;
  tableColumns: string[];
};

export type ValidateTrainingSetupParams = {
  selectedDatasetId: string | null;
  resolvedTargetColumn: string;
  excludeColumns: string[];
  dateColumns: string[];
  isLinearBaselineMode: boolean;
  validations: TrainingInputValidations;
};

export type ResolveTeacherRunKeyParams = {
  run: TrainingRunRow;
};

export type TrainingSweepValidations = {
  epochsValidation: ValidationResult<number>;
  testSizesValidation: ValidationResult<number>;
  learningRatesValidation: ValidationResult<number>;
  batchSizesValidation: ValidationResult<number>;
  hiddenDimsValidation: ValidationResult<number>;
  numHiddenLayersValidation: ValidationResult<number>;
  dropoutsValidation: ValidationResult<number>;
};

export type CalculatePlannedRunCountParams = {
  isLinearBaselineMode: boolean;
  validations: TrainingSweepValidations;
};

export type IsCompletedRunForModeParams = {
  run: TrainingRunRow;
  mode: string;
};

export type HasTeacherModelReferenceParams = {
  runId: string;
  modelId: string;
  modelPath: string;
};

export type OptimalPrediction = {
  metricName: string;
  metricValue: number;
};

export type MlTrainingUiBaseState = {
  targetColumn: string;
  setTargetColumn: (value: string) => void;
  excludeColumnsInput: string | null;
  setExcludeColumnsInput: (value: string | null) => void;
  dateColumnsInput: string | null;
  setDateColumnsInput: (value: string | null) => void;
  task: MlTaskType;
  setTask: (value: MlTaskType) => void;
  epochValuesInput: string;
  setEpochValuesInput: (value: string) => void;
  testSizesInput: string;
  setTestSizesInput: (value: string) => void;
  learningRatesInput: string;
  setLearningRatesInput: (value: string) => void;
  batchSizesInput: string;
  setBatchSizesInput: (value: string) => void;
  hiddenDimsInput: string;
  setHiddenDimsInput: (value: string) => void;
  numHiddenLayersInput: string;
  setNumHiddenLayersInput: (value: string) => void;
  dropoutsInput: string;
  setDropoutsInput: (value: string) => void;
  runSweepEnabled: boolean;
  setRunSweepEnabled: (value: boolean) => void;
  savedNumericInputs: NumericInputSnapshot | null;
  setSavedNumericInputs: (value: NumericInputSnapshot | null) => void;
  savedSweepInputs: NumericInputSnapshot | null;
  setSavedSweepInputs: (value: NumericInputSnapshot | null) => void;
  isTraining: boolean;
  setIsTraining: (value: boolean) => void;
  isDistilling: boolean;
  setIsDistilling: (value: boolean) => void;
  autoDistillEnabled: boolean;
  setAutoDistillEnabled: (value: boolean) => void;
  trainingProgress: TrainingProgress;
  setTrainingProgress: (value: TrainingProgress) => void;
  trainingError: string | null;
  setTrainingError: (value: string | null) => void;
  copyRunsStatus: string | null;
  setCopyRunsStatus: (value: string | null) => void;
  optimizerStatus: string | null;
  setOptimizerStatus: (value: string | null) => void;
  distillStatus: string | null;
  setDistillStatus: (value: string | null) => void;
  saveDistilledModel: boolean;
  setSaveDistilledModel: (value: boolean) => void;
  isOptimalModalOpen: boolean;
  setIsOptimalModalOpen: (value: boolean) => void;
  pendingOptimalParams: HyperParams | null;
  setPendingOptimalParams: (value: HyperParams | null) => void;
  pendingOptimalPrediction: OptimalPrediction | null;
  setPendingOptimalPrediction: (value: OptimalPrediction | null) => void;
  isDistillMetricsModalOpen: boolean;
  setIsDistillMetricsModalOpen: (value: boolean) => void;
  distillMetrics: TrainingMetrics | null;
  setDistillMetrics: (value: TrainingMetrics | null) => void;
  distillModelId: string | null;
  setDistillModelId: (value: string | null) => void;
  distillModelPath: string | null;
  setDistillModelPath: (value: string | null) => void;
  distillComparison: DistillComparison | null;
  setDistillComparison: (value: DistillComparison | null) => void;
};

export type FrameworkTrainingUiState<TMode extends string> =
  MlTrainingUiBaseState & {
    trainingMode: TMode;
    setTrainingMode: (value: TMode) => void;
  };

export type TrainingProblemBase = {
  datasetId: string;
  targetColumn: string;
  task: MlTaskType;
  trainingMode: string;
  isLinearBaselineMode: boolean;
  excludeColumns: string[];
  dateColumns: string[];
  combinations: SweepCombination[];
};

export type TrainModelRequestBase = {
  dataset_id: string;
  target_column: string;
  training_mode?: string;
  save_model?: boolean;
  exclude_columns?: string[];
  date_columns?: string[];
  task?: MlTaskType;
  epochs?: number;
  batch_size?: number;
  learning_rate?: number;
  test_size?: number;
  hidden_dim?: number;
  num_hidden_layers?: number;
  dropout?: number;
};

export type TrainModelResultBase = {
  status: "ok" | "error";
  run_id?: string;
  model_id?: string;
  model_path?: string;
  metrics?: unknown;
  error?: string;
};

export type TrainingDepsBase<
  TTrainRequest extends TrainModelRequestBase,
  TTrainResult extends TrainModelResultBase,
> = {
  trainModel: (payload: TTrainRequest) => Promise<TTrainResult>;
  prependTrainingRun: (row: TrainingRunRow) => void;
  onProgress: (current: number, total: number) => void;
  formatCompletedAt: (params: { date?: Date }, options?: Record<string, never>) => string;
  formatMetricNumber: (params: { value: unknown }, options?: Record<string, never>) => string;
  shouldContinue?: () => boolean;
};

export type DistillationProblemBase = {
  datasetId: string;
  targetColumn: string;
  task: MlTaskType;
  trainingMode: string;
  saveDistilledModel: boolean;
  excludeColumns: string[];
  dateColumns: string[];
  teacher: {
    hidden: number;
    layers: number;
    dropout: number;
    epochs: number;
    batch: number;
    learningRate: number;
    testSize: number;
    runId?: string;
    modelId?: string;
    modelPath?: string;
  };
};

export type DistillModelRequestBase = {
  dataset_id: string;
  target_column: string;
  training_mode?: string;
  save_model?: boolean;
  teacher_run_id?: string;
  teacher_model_id?: string;
  teacher_model_path?: string;
  exclude_columns?: string[];
  date_columns?: string[];
  task?: MlTaskType;
  epochs?: number;
  batch_size?: number;
  learning_rate?: number;
  test_size?: number;
  temperature?: number;
  alpha?: number;
  student_hidden_dim?: number;
  student_num_hidden_layers?: number;
  student_dropout?: number;
};

export type DistillModelResultBase = {
  status: "ok" | "error";
  model_id?: string;
  model_path?: string;
  run_id?: string;
  metrics?: unknown;
  teacher_input_dim?: number | null;
  teacher_output_dim?: number | null;
  student_input_dim?: number | null;
  student_output_dim?: number | null;
  teacher_model_size_bytes?: number | null;
  student_model_size_bytes?: number | null;
  size_saved_bytes?: number | null;
  size_saved_percent?: number | null;
  teacher_param_count?: number | null;
  student_param_count?: number | null;
  param_saved_count?: number | null;
  param_saved_percent?: number | null;
  error?: string;
};

export type DistillationDepsBase<
  TDistillRequest extends DistillModelRequestBase,
  TDistillResult extends DistillModelResultBase,
> = {
  distillModel: (payload: TDistillRequest) => Promise<TDistillResult>;
  formatCompletedAt: (params: { date?: Date }, options?: Record<string, never>) => string;
  formatMetricNumber: (params: { value: unknown }, options?: Record<string, never>) => string;
};

export type TrainingSweepResult = {
  stopped: boolean;
  completed: number;
  total: number;
  completedTeacherRuns: TrainingRunRow[];
  failedRuns: number;
  firstFailureMessage: string | null;
};

export type DistillationSuccessResult = {
  status: "ok";
  metrics: TrainingMetrics;
  modelId: string | null;
  modelPath: string | null;
  runId: string | null;
  teacherModelSizeBytes: number | null;
  studentModelSizeBytes: number | null;
  teacherInputDim: number | null;
  teacherOutputDim: number | null;
  studentInputDim: number | null;
  studentOutputDim: number | null;
  sizeSavedBytes: number | null;
  sizeSavedPercent: number | null;
  teacherParamCount: number | null;
  studentParamCount: number | null;
  paramSavedCount: number | null;
  paramSavedPercent: number | null;
  distilledRun: TrainingRunRow;
};

export type DistillationRunResult =
  | { status: "error"; error: string }
  | DistillationSuccessResult;

export type CommonTrainingOutcome = TrainingSweepResult;
export type CommonTeacherConfig = DistillationProblemBase["teacher"];
export type CommonDistillationOkResult = DistillationSuccessResult;
export type CommonDistillationResult = DistillationRunResult;

export type CommonTrainingProblem<TMode extends string> =
  Omit<TrainingProblemBase, "trainingMode"> & {
    trainingMode: TMode;
  };

export type CommonTrainingDeps<TTrainPayload, TTrainResult> = {
  trainModel: (payload: TTrainPayload) => Promise<TTrainResult>;
  prependTrainingRun: (row: TrainingRunRow) => void;
  onProgress: (current: number, total: number) => void;
  formatCompletedAt: (params: { date?: Date }, options?: Record<string, never>) => string;
  formatMetricNumber: (params: { value: unknown }, options?: Record<string, never>) => string;
  shouldContinue?: () => boolean;
};

export type CommonDistillationProblem<TMode extends string> =
  Omit<DistillationProblemBase, "trainingMode"> & {
    trainingMode: TMode;
  };

export type CommonDistillationDeps<TDistillPayload, TDistillResult> = {
  distillModel: (payload: TDistillPayload) => Promise<TDistillResult>;
  formatCompletedAt: (params: { date?: Date }, options?: Record<string, never>) => string;
  formatMetricNumber: (params: { value: unknown }, options?: Record<string, never>) => string;
};

export type RunCommonTrainingFn<
  TMode extends string,
  TTrainPayload,
  TTrainResult,
> = (
  problem: CommonTrainingProblem<TMode>,
  deps: CommonTrainingDeps<TTrainPayload, TTrainResult>
) => Promise<CommonTrainingOutcome>;

export type RunCommonDistillationFn<
  TMode extends string,
  TDistillPayload,
  TDistillResult,
> = (
  problem: CommonDistillationProblem<TMode>,
  deps: CommonDistillationDeps<TDistillPayload, TDistillResult>
) => Promise<CommonDistillationResult>;

export type FrameworkTrainingAdapters<
  TMode extends string,
  TTrainPayload,
  TTrainResult,
  TDistillPayload,
  TDistillResult,
> = {
  isDistillationSupportedMode: (mode: string) => mode is TMode;
  trainModel: (payload: TTrainPayload) => Promise<TTrainResult>;
  distillModel: (payload: TDistillPayload) => Promise<TDistillResult>;
};

export type UseTrainingFrameworkLogicArgs<
  TMode extends string,
  TTrainPayload,
  TTrainResult,
  TDistillPayload,
  TDistillResult,
> = {
  dataset: MlDatasetViewModel;
  trainingRuns: TrainingRunRow[];
  prependTrainingRun: (row: TrainingRunRow) => void;
  ui: FrameworkTrainingUiState<TMode>;
  runTraining: RunCommonTrainingFn<TMode, TTrainPayload, TTrainResult>;
  runDistillation: RunCommonDistillationFn<TMode, TDistillPayload, TDistillResult>;
  runtime?: Partial<BaseTrainingRuntimeDeps>;
  framework: FrameworkTrainingAdapters<
    TMode,
    TTrainPayload,
    TTrainResult,
    TDistillPayload,
    TDistillResult
  >;
};

export type FrameworkDistilledSnapshot = DistilledSnapshotByTeacher;

export type NumericInputState = NumericInputSnapshot &
  NumericInputSetters & {
    savedNumericInputs: NumericInputSnapshot | null;
    setSavedNumericInputs: (value: NumericInputSnapshot | null) => void;
    savedSweepInputs: NumericInputSnapshot | null;
    setSavedSweepInputs: (value: NumericInputSnapshot | null) => void;
    setRunSweepEnabled: (value: boolean) => void;
  };

export type OptimizerUiState = {
  pendingOptimalParams: HyperParams | null;
  setPendingOptimalParams: (value: HyperParams | null) => void;
  setPendingOptimalPrediction: (value: OptimalPrediction | null) => void;
  setIsOptimalModalOpen: (value: boolean) => void;
  setOptimizerStatus: (value: string | null) => void;
};

export type HandleFindOptimalParamsArgs = {
  trainingRuns: TrainingRunRow[];
  ui: OptimizerUiState;
};

export type TrainingSharedScheduler = (
  callback: () => void,
  delayMs: number
) => void;

export type TrainingSharedClipboardWriter = (text: string) => Promise<void>;

export type TrainingSharedRuntime = {
  schedule: TrainingSharedScheduler;
  writeClipboardText: TrainingSharedClipboardWriter;
};

export type HandleApplyOptimalParamsUi = OptimizerUiState &
  Pick<
    NumericInputSetters,
    | "setEpochValuesInput"
    | "setLearningRatesInput"
    | "setTestSizesInput"
    | "setBatchSizesInput"
    | "setHiddenDimsInput"
    | "setNumHiddenLayersInput"
    | "setDropoutsInput"
  > & {
    setRunSweepEnabled: (value: boolean) => void;
  };

export type HandleCopyTrainingRunsArgs = {
  trainingRuns: TrainingRunRow[];
  setCopyRunsStatus: (value: string | null) => void;
};
