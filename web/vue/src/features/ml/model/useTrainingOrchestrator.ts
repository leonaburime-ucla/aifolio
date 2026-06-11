import { ref, computed } from "vue";
import { createMlTrainingApi } from "../api";
import type { MlTrainingApi, TrainPayload, TrainResponse } from "../api";
import type {
  TrainingMetrics,
  TrainingRunRow as ContractTrainingRunRow,
} from "@aifolio/contracts/entities/ml-training";
import {
  formatCompletedAt,
  formatMetricNumber,
} from "@aifolio/frontend-core/ml-training";

export type Framework = "pytorch" | "tensorflow";

export type TrainingCombo = {
  epochs: number;
  batch_size: number;
  learning_rate: number;
  test_size: number;
  hidden_dim: number;
  num_hidden_layers: number;
  dropout: number;
};

export type TrainingRunRow = ContractTrainingRunRow;

export type UseTrainingOrchestratorOptions = {
  baseUrl: string;
  framework: Framework;
  api?: MlTrainingApi;
};

function toCompletedRunRow({
  combo,
  data,
  payload,
}: {
  combo: TrainingCombo;
  data: TrainResponse;
  payload: {
    datasetId: string;
    targetColumn: string;
    trainingMode: string;
    task: string;
  };
}): TrainingRunRow {
  const metrics = (data.metrics ?? {}) as TrainingMetrics & {
    test_metric?: number;
    accuracy?: number;
  };
  const metricValue =
    metrics.test_metric_value ?? metrics.test_metric ?? metrics.accuracy ?? null;
  const isLinearBaseline = payload.trainingMode === "linear_glm_baseline";

  return {
    result: "completed",
    status: "completed",
    completed_at: formatCompletedAt({}),
    epochs: combo.epochs,
    learning_rate: formatMetricNumber({ value: combo.learning_rate }),
    test_size: formatMetricNumber({ value: combo.test_size }),
    batch_size: combo.batch_size,
    hidden_dim: isLinearBaseline ? "n/a" : combo.hidden_dim,
    num_hidden_layers: isLinearBaseline ? "n/a" : combo.num_hidden_layers,
    dropout: isLinearBaseline ? "n/a" : formatMetricNumber({ value: combo.dropout }),
    task: payload.task,
    training_mode: payload.trainingMode,
    target_column: payload.targetColumn,
    dataset_id: payload.datasetId,
    metric_name:
      metrics.test_metric_name ?? (typeof metrics.accuracy === "number" ? "accuracy" : "n/a"),
    metric_score: formatMetricNumber({ value: metricValue }),
    metric: metricValue,
    train_loss: formatMetricNumber({ value: metrics.train_loss }),
    test_loss: formatMetricNumber({ value: metrics.test_loss }),
    model_id: data.model_id ?? "n/a",
    model_path: data.model_path ?? "n/a",
    run_id: data.run_id ?? "n/a",
    error: data.error ?? null,
  };
}

function toFailedRunRow({
  combo,
  payload,
  error,
}: {
  combo: TrainingCombo;
  payload: {
    datasetId: string;
    targetColumn: string;
    trainingMode: string;
    task: string;
  };
  error: string;
}): TrainingRunRow {
  const isLinearBaseline = payload.trainingMode === "linear_glm_baseline";

  return {
    result: "failed",
    status: "failed",
    completed_at: formatCompletedAt({}),
    epochs: combo.epochs,
    learning_rate: formatMetricNumber({ value: combo.learning_rate }),
    test_size: formatMetricNumber({ value: combo.test_size }),
    batch_size: combo.batch_size,
    hidden_dim: isLinearBaseline ? "n/a" : combo.hidden_dim,
    num_hidden_layers: isLinearBaseline ? "n/a" : combo.num_hidden_layers,
    dropout: isLinearBaseline ? "n/a" : formatMetricNumber({ value: combo.dropout }),
    task: payload.task,
    training_mode: payload.trainingMode,
    target_column: payload.targetColumn,
    dataset_id: payload.datasetId,
    metric_name: "n/a",
    metric_score: "n/a",
    metric: null,
    train_loss: "n/a",
    test_loss: "n/a",
    model_id: "n/a",
    model_path: "n/a",
    run_id: "n/a",
    error,
  };
}

export function useTrainingOrchestrator(options: UseTrainingOrchestratorOptions) {
  const api = options.api ?? createMlTrainingApi({ baseUrl: options.baseUrl });
  const trainFn = options.framework === "pytorch" ? api.trainPytorch : api.trainTensorflow;

  const isTraining = ref(false);
  const stopTraining = ref(false);
  const trainingError = ref<string | null>(null);
  const trainingRuns = ref<TrainingRunRow[]>([]);
  const trainingProgress = ref({ current: 0, total: 0 });

  function buildCombos(params: {
    epochs: number[];
    batches: number[];
    learningRates: number[];
    testSizes: number[];
    hiddenDims: number[];
    numHiddenLayers: number[];
    dropouts: number[];
  }) {
    const combos: TrainingCombo[] = [];
    for (const epochs of params.epochs) {
      for (const batch_size of params.batches) {
        for (const learning_rate of params.learningRates) {
          for (const test_size of params.testSizes) {
            combos.push({
              epochs,
              batch_size,
              learning_rate,
              test_size,
              hidden_dim: params.hiddenDims[0] ?? 128,
              num_hidden_layers: params.numHiddenLayers[0] ?? 2,
              dropout: params.dropouts[0] ?? 0.1,
            });
          }
        }
      }
    }
    return combos;
  }

  const plannedRunCount = computed(() => trainingProgress.value.total);

  async function train(payload: {
    datasetId: string;
    targetColumn: string;
    trainingMode: string;
    task: string;
    excludeColumns: string[];
    dateColumns: string[];
    combos: TrainingCombo[];
  }) {
    isTraining.value = true;
    stopTraining.value = false;
    trainingError.value = null;
    trainingRuns.value = [];
    trainingProgress.value = { current: 0, total: payload.combos.length };

    for (const combo of payload.combos) {
      if (stopTraining.value) break;
      trainingProgress.value.current++;

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
        const data: TrainResponse = await trainFn(request);
        trainingRuns.value.push(
          data.status === "ok"
            ? toCompletedRunRow({ combo, data, payload })
            : toFailedRunRow({
                combo,
                payload,
                error: data.error ?? "Training failed.",
              })
        );
      } catch (err) {
        trainingRuns.value.push(
          toFailedRunRow({
            combo,
            payload,
            error: err instanceof Error ? err.message : "Request failed",
          })
        );
      }
    }

    isTraining.value = false;
  }

  function stop() {
    stopTraining.value = true;
  }

  function clearRuns() {
    trainingRuns.value = [];
  }

  function copyResults(): string {
    return JSON.stringify(trainingRuns.value, null, 2);
  }

  return {
    isTraining,
    stopTraining,
    trainingError,
    trainingRuns,
    trainingProgress,
    plannedRunCount,
    buildCombos,
    train,
    stop,
    clearRuns,
    copyResults,
  };
}
