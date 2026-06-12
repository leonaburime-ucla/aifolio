import { Injectable, computed, inject, signal } from '@angular/core';
import {
  buildDistillActionModel,
  buildDistillationComparison,
  buildEnrichedDistilledRun,
  calculatePlannedRunCount,
  createReloadSweepValuesHandler,
  createToggleRunSweepHandler,
  formatCompletedAt,
  formatMetricNumber,
  getPytorchModeExplainer,
  getTensorflowModeExplainer,
  getTrainingDefaults,
  handleApplyOptimalParams,
  handleCopyTrainingRuns,
  handleFindOptimalParams,
  resolveDefaultTrainingDatasetId,
  resolveDistilledModalPayload,
  resolveTeacherRunKey,
  splitColumnInput,
  validateBatchSizes,
  validateDropouts,
  validateEpochValues,
  validateHiddenDims,
  validateLearningRates,
  validateNumHiddenLayers,
  validateTestSizes,
  validateTrainingSetup,
} from '@aifolio/frontend-core/ml-training';
import type { TrainingMetrics, TrainingRunRow } from '@aifolio/contracts/entities/ml-training';
import type { DatasetOption } from '../../../shared/types/dataset-option';
import { MlTrainingApiService } from '../api/ml-training-api.service';
import type { DistillPayload, DistillResponse, Framework, TrainPayload, TrainResponse, TrainingCombo, TrainingRow } from '../model/ml-training.types';

type TrainingConfig = {
  framework: Framework;
  defaultTrainingMode: string;
  defaultExcludeColumns?: string;
};

function numericTeacherValue(value: unknown, fallback: number): number {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

function nullableTeacherReference(value: unknown): string | undefined {
  const text = String(value ?? '').trim();
  return text && text !== 'n/a' ? text : undefined;
}

function resolveDistilledEpochs(framework: Framework, teacherEpochs: number): number {
  if (framework === 'tensorflow') return Math.min(24, Math.max(8, Math.round(teacherEpochs * 0.4)));
  return Math.max(30, Math.round(teacherEpochs));
}

function validationValues(result: { ok: boolean; values?: number[] }, fallback: number[]): number[] {
  return result.ok && Array.isArray(result.values) ? result.values : fallback;
}

function completedRow(input: {
  combo: TrainingCombo;
  data: TrainResponse;
  payload: { datasetId: string; targetColumn: string; trainingMode: string; task: string };
}): TrainingRunRow {
  const metrics = (input.data.metrics ?? {}) as TrainingMetrics & { test_metric?: number; accuracy?: number };
  const metricValue = metrics.test_metric_value ?? metrics.test_metric ?? metrics.accuracy ?? null;
  const isLinearBaseline = input.payload.trainingMode === 'linear_glm_baseline';
  return {
    result: 'completed',
    status: 'completed',
    completed_at: formatCompletedAt({}),
    epochs: input.combo.epochs,
    learning_rate: formatMetricNumber({ value: input.combo.learning_rate }),
    test_size: formatMetricNumber({ value: input.combo.test_size }),
    batch_size: input.combo.batch_size,
    hidden_dim: isLinearBaseline ? 'n/a' : input.combo.hidden_dim,
    num_hidden_layers: isLinearBaseline ? 'n/a' : input.combo.num_hidden_layers,
    dropout: isLinearBaseline ? 'n/a' : formatMetricNumber({ value: input.combo.dropout }),
    task: input.payload.task,
    training_mode: input.payload.trainingMode,
    target_column: input.payload.targetColumn,
    dataset_id: input.payload.datasetId,
    metric_name: metrics.test_metric_name ?? (typeof metrics.accuracy === 'number' ? 'accuracy' : 'n/a'),
    metric_score: formatMetricNumber({ value: metricValue }),
    metric: metricValue,
    train_loss: formatMetricNumber({ value: metrics.train_loss }),
    test_loss: formatMetricNumber({ value: metrics.test_loss }),
    model_id: input.data.model_id ?? 'n/a',
    model_path: input.data.model_path ?? 'n/a',
    run_id: input.data.run_id ?? 'n/a',
    error: input.data.error ?? null,
  };
}

function failedRow(input: {
  combo: TrainingCombo;
  payload: { datasetId: string; targetColumn: string; trainingMode: string; task: string };
  error: string;
}): TrainingRunRow {
  const isLinearBaseline = input.payload.trainingMode === 'linear_glm_baseline';
  return {
    result: 'failed',
    status: 'failed',
    completed_at: formatCompletedAt({}),
    epochs: input.combo.epochs,
    learning_rate: formatMetricNumber({ value: input.combo.learning_rate }),
    test_size: formatMetricNumber({ value: input.combo.test_size }),
    batch_size: input.combo.batch_size,
    hidden_dim: isLinearBaseline ? 'n/a' : input.combo.hidden_dim,
    num_hidden_layers: isLinearBaseline ? 'n/a' : input.combo.num_hidden_layers,
    dropout: isLinearBaseline ? 'n/a' : formatMetricNumber({ value: input.combo.dropout }),
    task: input.payload.task,
    training_mode: input.payload.trainingMode,
    target_column: input.payload.targetColumn,
    dataset_id: input.payload.datasetId,
    metric_name: 'n/a',
    metric_score: 'n/a',
    metric: null,
    train_loss: 'n/a',
    test_loss: 'n/a',
    model_id: 'n/a',
    model_path: 'n/a',
    run_id: 'n/a',
    error: input.error,
  };
}

function fallbackDistilledRun(result: DistillResponse, payload: DistillPayload): TrainingRunRow {
  const metrics = result.metrics as TrainingMetrics | undefined;
  return {
    result: 'distilled',
    completed_at: new Date().toISOString(),
    epochs: payload.epochs,
    learning_rate: payload.learning_rate.toString(),
    test_size: payload.test_size.toString(),
    batch_size: payload.batch_size,
    hidden_dim: payload.student_hidden_dim,
    num_hidden_layers: payload.student_num_hidden_layers,
    dropout: payload.student_dropout.toString(),
    task: payload.task,
    training_mode: payload.training_mode,
    target_column: payload.target_column,
    dataset_id: payload.dataset_id,
    metric_name: metrics?.test_metric_name ?? 'n/a',
    metric_score: String(metrics?.test_metric_value ?? 'n/a'),
    train_loss: String(metrics?.train_loss ?? 'n/a'),
    test_loss: String(metrics?.test_loss ?? 'n/a'),
    model_id: result.model_id ?? 'n/a',
    model_path: result.model_path ?? 'n/a',
    run_id: result.run_id ?? 'n/a',
  };
}

@Injectable()
export class TrainingScreenOrchestrator {
  private readonly api = inject(MlTrainingApiService);
  private configuredFramework: Framework | null = null;
  private datasetLoadRequestId = 0;
  private readonly snapshotsByTeacher = signal<Record<string, unknown>>({});

  readonly baseUrl = signal('/api/ai');
  readonly framework = signal<Framework>('pytorch');
  readonly datasetOptions = signal<DatasetOption[]>([]);
  readonly selectedDatasetId = signal<string | null>(null);
  readonly tableRows = signal<Record<string, unknown>[]>([]);
  readonly tableColumns = signal<string[]>([]);
  readonly targetColumn = signal('');
  readonly datasetError = signal<string | null>(null);

  readonly isTraining = signal(false);
  readonly stopTraining = signal(false);
  readonly trainingError = signal<string | null>(null);
  readonly trainingRuns = signal<TrainingRow[]>([]);
  readonly trainingProgress = signal({ current: 0, total: 0 });

  readonly trainingMode = signal('linear_glm_baseline');
  readonly task = signal('auto');
  readonly epochValues = signal('60');
  readonly batchSizes = signal('64');
  readonly learningRates = signal('0.001');
  readonly testSizes = signal('0.2');
  readonly hiddenDims = signal('128');
  readonly numHiddenLayers = signal('2');
  readonly dropouts = signal('0.1');
  readonly excludeColumns = signal('');
  readonly dateColumns = signal('');
  readonly sweepEnabled = signal(false);
  readonly autoDistill = signal(false);

  readonly isModelPreviewOpen = signal(false);
  readonly isOptimalModalOpen = signal(false);
  readonly pendingOptimalParams = signal<any>(null);
  readonly pendingOptimalPrediction = signal<any>(null);
  readonly optimizerStatus = signal<string | null>(null);
  readonly isDistillMetricsModalOpen = signal(false);
  readonly distillMetrics = signal<any>(null);
  readonly distillModelId = signal<string | null>(null);
  readonly distillModelPath = signal<string | null>(null);
  readonly distillComparison = signal<any>(null);
  readonly distillingTeacherKey = signal<string | null>(null);
  readonly distilledByTeacher = signal<Record<string, string>>({});
  readonly distillStatus = signal<string | null>(null);
  readonly copyRunsStatus = signal<string | null>(null);

  readonly isLinearBaseline = computed(() => this.trainingMode() === 'linear_glm_baseline');
  readonly epochsValidation = computed(() => validateEpochValues({ raw: this.epochValues() }));
  readonly batchSizesValidation = computed(() => validateBatchSizes({ raw: this.batchSizes() }));
  readonly learningRatesValidation = computed(() => validateLearningRates({ raw: this.learningRates() }));
  readonly testSizesValidation = computed(() => validateTestSizes({ raw: this.testSizes() }));
  readonly hiddenDimsValidation = computed(() => validateHiddenDims({ raw: this.hiddenDims() }));
  readonly numHiddenLayersValidation = computed(() => validateNumHiddenLayers({ raw: this.numHiddenLayers() }));
  readonly dropoutsValidation = computed(() => validateDropouts({ raw: this.dropouts() }));
  readonly defaults = computed(() => getTrainingDefaults(this.selectedDatasetId()));
  readonly modeExplainer = computed(() =>
    this.framework() === 'pytorch'
      ? getPytorchModeExplainer(this.trainingMode())
      : getTensorflowModeExplainer(this.trainingMode())
  );
  readonly plannedRunCount = computed(() =>
    calculatePlannedRunCount({
      isLinearBaselineMode: this.isLinearBaseline(),
      validations: this.validationSnapshot(),
    })
  );
  readonly isTrainDisabled = computed(() => this.isTraining() || !this.selectedDatasetId() || this.plannedRunCount() === 0);
  readonly completedRuns = computed(() => this.trainingRuns().filter((run) => run.result === 'completed' || run.status === 'completed'));

  configure(config: TrainingConfig): void {
    if (this.configuredFramework === config.framework) return;
    this.configuredFramework = config.framework;
    this.framework.set(config.framework);
    this.trainingMode.set(config.defaultTrainingMode);
    this.excludeColumns.set(config.defaultExcludeColumns ?? '');
    this.resetTrainingState();
    void this.loadManifest();
  }

  async loadManifest(): Promise<void> {
    try {
      const { datasets } = await this.api.fetchManifest(this.baseUrl());
      const options = datasets.map((dataset) => ({ id: dataset.id, label: dataset.label ?? dataset.id }));
      this.datasetOptions.set(options);
      const resolved = resolveDefaultTrainingDatasetId({
        selectedDatasetId: this.selectedDatasetId(),
        datasets: options,
      });
      if (resolved && !this.selectedDatasetId()) await this.onDatasetChange(resolved);
    } catch (err) {
      this.datasetError.set(err instanceof Error ? err.message : 'Error');
    }
  }

  async onDatasetChange(id: string): Promise<void> {
    this.selectedDatasetId.set(id);
    const defaults = getTrainingDefaults(id);
    this.targetColumn.set(defaults.targetColumn);
    this.excludeColumns.set(defaults.excludeColumns.join(','));
    this.dateColumns.set(defaults.dateColumns.join(','));
    this.task.set(defaults.task);
    this.epochValues.set(String(defaults.epochs));
    this.testSizes.set('0.2');
    this.learningRates.set('0.001');
    this.batchSizes.set('64');
    this.hiddenDims.set('128');
    this.numHiddenLayers.set('2');
    this.dropouts.set('0.1');
    this.sweepEnabled.set(false);
    this.trainingError.set(null);
    await this.loadDataset(id);
  }

  async loadDataset(id: string): Promise<void> {
    const requestId = ++this.datasetLoadRequestId;
    this.datasetError.set(null);
    this.tableRows.set([]);
    this.tableColumns.set([]);
    try {
      const payload = await this.api.fetchDataset(this.baseUrl(), id);
      if (requestId !== this.datasetLoadRequestId || id !== this.selectedDatasetId()) return;
      const rows = payload.rows ?? [];
      const columns = payload.columns ?? (rows.length > 0 ? Object.keys(rows[0]) : []);
      this.tableRows.set(rows);
      this.tableColumns.set(columns);
      if (!this.targetColumn() && columns.length > 0) {
        this.targetColumn.set(columns[columns.length - 1]);
      }
    } catch (err) {
      if (requestId !== this.datasetLoadRequestId || id !== this.selectedDatasetId()) return;
      this.datasetError.set(err instanceof Error ? err.message : 'Error');
    }
  }

  buildCombos(input: {
    epochs: number[];
    batches: number[];
    learningRates: number[];
    testSizes: number[];
    hiddenDims: number[];
    numHiddenLayers: number[];
    dropouts: number[];
  }): TrainingCombo[] {
    const combos: TrainingCombo[] = [];
    for (const epochs of input.epochs) {
      for (const batch_size of input.batches) {
        for (const learning_rate of input.learningRates) {
          for (const test_size of input.testSizes) {
            combos.push({
              epochs,
              batch_size,
              learning_rate,
              test_size,
              hidden_dim: input.hiddenDims[0] ?? 128,
              num_hidden_layers: input.numHiddenLayers[0] ?? 2,
              dropout: input.dropouts[0] ?? 0.1,
            });
          }
        }
      }
    }
    return combos;
  }

  async onTrain(): Promise<void> {
    const epochs = validationValues(this.epochsValidation(), [60]);
    const batches = validationValues(this.batchSizesValidation(), [64]);
    const lrs = validationValues(this.learningRatesValidation(), [0.001]);
    const tests = validationValues(this.testSizesValidation(), [0.2]);
    const dims = validationValues(this.hiddenDimsValidation(), [128]);
    const layers = validationValues(this.numHiddenLayersValidation(), [2]);
    const drops = validationValues(this.dropoutsValidation(), [0.1]);
    const targetCol = this.targetColumn() || this.defaults().targetColumn;
    const excludeColumns = splitColumnInput({ value: this.excludeColumns() });
    const dateColumns = splitColumnInput({ value: this.dateColumns() });

    const setupError = validateTrainingSetup({
      selectedDatasetId: this.selectedDatasetId(),
      resolvedTargetColumn: targetCol,
      excludeColumns,
      dateColumns,
      isLinearBaselineMode: this.isLinearBaseline(),
      validations: this.validationSnapshot(),
    });

    if (setupError) {
      this.trainingError.set(setupError);
      return;
    }

    const combos = this.buildCombos({
      epochs,
      batches,
      learningRates: lrs,
      testSizes: tests,
      hiddenDims: dims,
      numHiddenLayers: layers,
      dropouts: drops,
    });

    await this.train({
      datasetId: this.selectedDatasetId()!,
      targetColumn: targetCol,
      trainingMode: this.trainingMode(),
      task: this.task(),
      excludeColumns,
      dateColumns,
      combos,
    });

    if (this.autoDistill() && this.trainingRuns().length > 0) {
      if (!this.isDistillationSupportedForRun({ training_mode: this.trainingMode() })) {
        this.distillStatus.set(`Auto-distill skipped: '${this.trainingMode()}' distillation is not supported yet.`);
        window.setTimeout(() => this.distillStatus.set(null), 3500);
        return;
      }
      for (const run of this.trainingRuns()) {
        if (run.result === 'completed' || run.status === 'completed') await this.onDistillFromRun(run);
      }
    }
  }

  async train(payload: {
    datasetId: string;
    targetColumn: string;
    trainingMode: string;
    task: string;
    excludeColumns: string[];
    dateColumns: string[];
    combos: TrainingCombo[];
  }): Promise<void> {
    this.isTraining.set(true);
    this.stopTraining.set(false);
    this.trainingError.set(null);
    this.trainingRuns.set([]);
    this.trainingProgress.set({ current: 0, total: payload.combos.length });

    for (const combo of payload.combos) {
      if (this.stopTraining()) break;
      this.trainingProgress.update((progress) => ({ ...progress, current: progress.current + 1 }));
      const request: TrainPayload = {
        dataset_id: payload.datasetId,
        target_column: payload.targetColumn,
        training_mode: payload.trainingMode,
        task: payload.task,
        epochs: combo.epochs,
        batch_size: combo.batch_size,
        learning_rate: combo.learning_rate,
        test_size: combo.test_size,
        hidden_dim: combo.hidden_dim,
        num_hidden_layers: combo.num_hidden_layers,
        dropout: combo.dropout,
        exclude_columns: payload.excludeColumns,
        date_columns: payload.dateColumns,
      };

      try {
        const data =
          this.framework() === 'pytorch'
            ? await this.api.trainPytorch(this.baseUrl(), request)
            : await this.api.trainTensorflow(this.baseUrl(), request);
        this.trainingRuns.update((runs) => [
          ...runs,
          data.status === 'ok'
            ? completedRow({ combo, data, payload })
            : failedRow({ combo, payload, error: data.error ?? 'Training failed.' }),
        ]);
      } catch (err) {
        this.trainingRuns.update((runs) => [
          ...runs,
          failedRow({ combo, payload, error: err instanceof Error ? err.message : 'Request failed' }),
        ]);
      }
    }

    this.isTraining.set(false);
  }

  stop(): void {
    this.stopTraining.set(true);
  }

  clearRuns(): void {
    this.trainingRuns.set([]);
  }

  toggleRunSweep(enabled: boolean): void {
    const ui = this.sweepUiAdapter();
    createToggleRunSweepHandler({ ui, defaultEpochs: this.defaults().epochs ?? 60 })(enabled);
  }

  reloadSweepValues(): void {
    createReloadSweepValuesHandler({ ui: this.sweepUiAdapter() })();
  }

  onFindOptimalParamsClick(): void {
    handleFindOptimalParams(
      { trainingRuns: this.completedRuns() as any, ui: this.optimizerUiAdapter() },
      { runtime: this.runtimeAdapter() }
    );
  }

  onApplyOptimalParams(): void {
    handleApplyOptimalParams(
      { ui: this.optimizerUiAdapter() },
      { runtime: this.runtimeAdapter() }
    );
  }

  async onCopyResults(): Promise<void> {
    await handleCopyTrainingRuns(
      {
        trainingRuns: this.trainingRuns() as any,
        setCopyRunsStatus: (status: string | null) => this.copyRunsStatus.set(status),
      },
      { runtime: this.runtimeAdapter() }
    );
  }

  isDistillationSupportedForRun(run: Partial<TrainingRunRow>): boolean {
    const supported =
      this.framework() === 'tensorflow'
        ? ['mlp_dense', 'linear_glm_baseline', 'wide_and_deep']
        : ['mlp_dense', 'linear_glm_baseline', 'tabresnet'];
    return supported.includes(String(run.training_mode ?? ''));
  }

  distillAction(row: TrainingRunRow) {
    return buildDistillActionModel({
      row,
      isDistillationSupportedForRun: (candidate) => this.isDistillationSupportedForRun(candidate),
      distillingTeacherKey: this.distillingTeacherKey(),
      distilledByTeacher: this.distilledByTeacher(),
    });
  }

  async onDistillFromRun(run: TrainingRunRow): Promise<void> {
    if (!this.selectedDatasetId()) return;
    const teacherKey = resolveTeacherRunKey({ run });
    this.distillingTeacherKey.set(teacherKey);
    this.distillStatus.set('Running distillation...');
    this.isTraining.set(true);

    try {
      const teacherRunId = nullableTeacherReference(run.run_id);
      const teacherModelId = nullableTeacherReference(run.model_id);
      const teacherModelPath = nullableTeacherReference(run.model_path);
      if (!teacherRunId && !teacherModelId && !teacherModelPath) {
        this.trainingError.set('This run has no teacher model reference to distill from.');
        this.distillStatus.set('Distillation failed.');
        return;
      }

      const payload: DistillPayload = {
        dataset_id: this.selectedDatasetId()!,
        target_column: String(run.target_column || this.targetColumn()),
        training_mode: String(run.training_mode || this.trainingMode()),
        save_model: false,
        teacher_run_id: teacherRunId,
        teacher_model_id: teacherModelId,
        teacher_model_path: teacherModelPath,
        exclude_columns: splitColumnInput({ value: this.excludeColumns() }),
        date_columns: splitColumnInput({ value: this.dateColumns() }),
        task: String(run.task || this.task()),
        epochs: resolveDistilledEpochs(this.framework(), numericTeacherValue(run.epochs, 60)),
        batch_size: Math.max(1, Math.round(numericTeacherValue(run.batch_size, 64))),
        learning_rate: numericTeacherValue(run.learning_rate, 0.001),
        test_size: numericTeacherValue(run.test_size, 0.2),
        hidden_dim: numericTeacherValue(run.hidden_dim, 128),
        num_hidden_layers: numericTeacherValue(run.num_hidden_layers, 2),
        dropout: numericTeacherValue(run.dropout, 0.1),
        temperature: 2.5,
        alpha: 0.5,
        student_hidden_dim: Math.max(16, Math.round(numericTeacherValue(run.hidden_dim, 128) / 2)),
        student_num_hidden_layers: Math.max(1, Math.min(15, Math.round(numericTeacherValue(run.num_hidden_layers, 2) - 1))),
        student_dropout: Math.min(0.5, numericTeacherValue(run.dropout, 0.1) + 0.05),
      };

      const result =
        this.framework() === 'pytorch'
          ? await this.api.distillPytorch(this.baseUrl(), payload)
          : await this.api.distillTensorflow(this.baseUrl(), payload);

      if (result.status !== 'ok') {
        this.trainingError.set(result.error ?? 'Distillation failed.');
        this.distillStatus.set('Distillation failed.');
        return;
      }

      const distilledRun = fallbackDistilledRun(result, payload);
      const comparisonInput = {
        metrics: result.metrics ?? {},
        distilledRun,
        teacherInputDim: result.teacher_input_dim ?? null,
        teacherOutputDim: result.teacher_output_dim ?? null,
        studentInputDim: result.student_input_dim ?? null,
        studentOutputDim: result.student_output_dim ?? null,
        teacherModelSizeBytes: result.teacher_model_size_bytes ?? null,
        studentModelSizeBytes: result.student_model_size_bytes ?? null,
        sizeSavedBytes: result.size_saved_bytes ?? null,
        sizeSavedPercent: result.size_saved_percent ?? null,
        teacherParamCount: result.teacher_param_count ?? null,
        studentParamCount: result.student_param_count ?? null,
        paramSavedCount: result.param_saved_count ?? null,
        paramSavedPercent: result.param_saved_percent ?? null,
      };
      const { comparison, teacherMetricName, teacherMetricValue, studentMetricValue, qualityDelta } =
        buildDistillationComparison({ teacher: run as any, result: comparisonInput as any });
      const enriched = buildEnrichedDistilledRun({
        distilledRun,
        teacherKey,
        comparison,
        teacherMetricName,
        teacherMetricValue,
        studentMetricValue,
        qualityDelta,
      });

      this.distillMetrics.set(result.metrics);
      this.distillModelId.set(result.model_id ?? result.run_id ?? null);
      this.distillModelPath.set(result.model_path ?? null);
      this.distillComparison.set(comparison);
      this.isDistillMetricsModalOpen.set(true);
      this.trainingRuns.update((runs) => [enriched as TrainingRunRow, ...runs]);
      this.distilledByTeacher.update((current) => ({
        ...current,
        [teacherKey]: result.run_id ?? result.model_id ?? result.model_path ?? 'ready',
      }));
      this.snapshotsByTeacher.update((current) => ({
        ...current,
        [teacherKey]: { metrics: result.metrics, modelId: result.model_id, modelPath: result.model_path, comparison },
      }));
      this.distillStatus.set('Distilled student model created.');
      window.setTimeout(() => this.distillStatus.set(null), 2500);
    } catch (err) {
      this.trainingError.set(err instanceof Error ? err.message : 'Distillation failed.');
      this.distillStatus.set('Distillation failed.');
    } finally {
      this.isTraining.set(false);
      this.distillingTeacherKey.set(null);
    }
  }

  onSeeDistilledFromRun(run: TrainingRunRow): void {
    const teacherKey = resolveTeacherRunKey({ run });
    const payload = resolveDistilledModalPayload({
      teacherKey,
      snapshotsByTeacher: this.snapshotsByTeacher() as any,
      trainingRuns: this.trainingRuns() as any,
    });
    if (payload.status === 'missing') {
      this.trainingError.set('No distilled result found yet for this teacher run.');
      return;
    }
    this.distillMetrics.set(payload.metrics);
    this.distillModelId.set(payload.modelId);
    this.distillModelPath.set(payload.modelPath);
    this.distillComparison.set(payload.comparison);
    this.isDistillMetricsModalOpen.set(true);
  }

  private resetTrainingState(): void {
    this.trainingRuns.set([]);
    this.trainingError.set(null);
    this.distilledByTeacher.set({});
    this.snapshotsByTeacher.set({});
    this.distillStatus.set(null);
    this.copyRunsStatus.set(null);
  }

  private validationSnapshot() {
    return {
      epochsValidation: this.epochsValidation(),
      testSizesValidation: this.testSizesValidation(),
      learningRatesValidation: this.learningRatesValidation(),
      batchSizesValidation: this.batchSizesValidation(),
      hiddenDimsValidation: this.hiddenDimsValidation(),
      numHiddenLayersValidation: this.numHiddenLayersValidation(),
      dropoutsValidation: this.dropoutsValidation(),
    };
  }

  private sweepUiAdapter() {
    const thisRef = this;
    return {
      get epochValuesInput() { return thisRef.epochValues(); },
      setEpochValuesInput: (value: string) => this.epochValues.set(value),
      get batchSizesInput() { return thisRef.batchSizes(); },
      setBatchSizesInput: (value: string) => this.batchSizes.set(value),
      get learningRatesInput() { return thisRef.learningRates(); },
      setLearningRatesInput: (value: string) => this.learningRates.set(value),
      get testSizesInput() { return thisRef.testSizes(); },
      setTestSizesInput: (value: string) => this.testSizes.set(value),
      get hiddenDimsInput() { return thisRef.hiddenDims(); },
      setHiddenDimsInput: (value: string) => this.hiddenDims.set(value),
      get numHiddenLayersInput() { return thisRef.numHiddenLayers(); },
      setNumHiddenLayersInput: (value: string) => this.numHiddenLayers.set(value),
      get dropoutsInput() { return thisRef.dropouts(); },
      setDropoutsInput: (value: string) => this.dropouts.set(value),
      get runSweepEnabled() { return thisRef.sweepEnabled(); },
      setRunSweepEnabled: (value: boolean) => this.sweepEnabled.set(value),
      get savedSweepInputs() { return null; },
      setSavedSweepInputs: (_value: unknown) => {},
      get savedNumericInputs() { return null; },
      setSavedNumericInputs: (_value: unknown) => {},
    } as any;
  }

  private optimizerUiAdapter() {
    return {
      ...this.sweepUiAdapter(),
      setOptimizerStatus: (message: string | null) => this.optimizerStatus.set(message),
      setPendingOptimalParams: (params: any) => this.pendingOptimalParams.set(params),
      setPendingOptimalPrediction: (prediction: any) => this.pendingOptimalPrediction.set(prediction),
      setIsOptimalModalOpen: (open: boolean) => this.isOptimalModalOpen.set(open),
    } as any;
  }

  private runtimeAdapter() {
    return {
      schedule: (callback: () => void, delayMs: number) => window.setTimeout(callback, delayMs),
      writeClipboardText: async (text: string) => {
        await navigator.clipboard.writeText(text);
      },
    };
  }
}
