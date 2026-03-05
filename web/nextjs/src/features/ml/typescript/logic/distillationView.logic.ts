import {
  type DistillComparison,
  type TrainingMetrics,
  type TrainingRunRow,
} from "@/features/ml/typescript/utils/trainingRuns.util";
import {
  metricHigherIsBetter,
  parseNumericValue,
} from "@/features/ml/typescript/utils/trainingUiShared";

type DistilledSnapshot = {
  metrics: TrainingMetrics;
  modelId: string | null;
  modelPath: string | null;
  comparison: DistillComparison;
};

/**
 * Builds teacher/student comparison metrics from a teacher run and distillation result.
 * @param params - Required parameters.
 * @returns Comparison payload plus teacher/student metric derivations.
 */
export function buildDistillationComparison(
  {
    teacher,
    result,
  }: {
    teacher: TrainingRunRow;
    result: {
      metrics: TrainingMetrics;
      distilledRun: TrainingRunRow;
      teacherInputDim: number | null;
      teacherOutputDim: number | null;
      studentInputDim: number | null;
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
  },
  {}: Record<string, never> = {}
): {
  comparison: DistillComparison;
  teacherMetricName: string;
  teacherMetricValue: number | null;
  studentMetricValue: number | null;
  qualityDelta: number | null;
} {
  const teacherMetricName = String(
    teacher.metric_name ?? result.metrics.test_metric_name ?? "accuracy"
  );
  const teacherMetricValue = parseNumericValue({ value: teacher.metric_score });
  const studentMetricValue =
    typeof result.metrics.test_metric_value === "number"
      ? result.metrics.test_metric_value
      : null;
  const higherIsBetter = metricHigherIsBetter({ metricName: teacherMetricName });
  const qualityDelta =
    teacherMetricValue !== null && studentMetricValue !== null
      ? higherIsBetter
        ? studentMetricValue - teacherMetricValue
        : teacherMetricValue - studentMetricValue
      : null;

  const comparison: DistillComparison = {
    metricName: teacherMetricName,
    teacherMetricValue,
    studentMetricValue,
    qualityDelta,
    higherIsBetter,
    teacherTrainingMode: String(teacher.training_mode ?? "") || null,
    studentTrainingMode: String(result.distilledRun.training_mode ?? "") || null,
    teacherHiddenDim: parseNumericValue({ value: teacher.hidden_dim }),
    studentHiddenDim: parseNumericValue({ value: result.distilledRun.hidden_dim }),
    teacherNumHiddenLayers: parseNumericValue({ value: teacher.num_hidden_layers }),
    studentNumHiddenLayers: parseNumericValue({
      value: result.distilledRun.num_hidden_layers,
    }),
    teacherInputDim: result.teacherInputDim,
    studentInputDim: result.studentInputDim,
    teacherOutputDim: result.teacherOutputDim,
    studentOutputDim: result.studentOutputDim,
    teacherModelSizeBytes: result.teacherModelSizeBytes,
    studentModelSizeBytes: result.studentModelSizeBytes,
    sizeSavedBytes: result.sizeSavedBytes,
    sizeSavedPercent: result.sizeSavedPercent,
    teacherParamCount: result.teacherParamCount,
    studentParamCount: result.studentParamCount,
    paramSavedCount: result.paramSavedCount,
    paramSavedPercent: result.paramSavedPercent,
  };

  return {
    comparison,
    teacherMetricName,
    teacherMetricValue,
    studentMetricValue,
    qualityDelta,
  };
}

/**
 * Adds distillation metadata fields onto the persisted distilled run row.
 * @param params - Required parameters.
 * @returns Enriched run row with distillation lineage metadata.
 */
export function buildEnrichedDistilledRun(
  {
    distilledRun,
    teacherKey,
    comparison,
    teacherMetricName,
    teacherMetricValue,
    studentMetricValue,
    qualityDelta,
  }: {
    distilledRun: TrainingRunRow;
    teacherKey: string;
    comparison: DistillComparison;
    teacherMetricName: string;
    teacherMetricValue: number | null;
    studentMetricValue: number | null;
    qualityDelta: number | null;
  },
  {}: Record<string, never> = {}
): TrainingRunRow {
  return {
    ...distilledRun,
    teacher_ref_key: teacherKey,
    distill_teacher_metric_name: teacherMetricName,
    distill_teacher_metric_value: teacherMetricValue ?? "n/a",
    distill_student_metric_value: studentMetricValue ?? "n/a",
    distill_quality_delta: qualityDelta ?? "n/a",
    distill_higher_is_better: comparison.higherIsBetter ? "1" : "0",
    distill_teacher_training_mode: comparison.teacherTrainingMode ?? "n/a",
    distill_student_training_mode: comparison.studentTrainingMode ?? "n/a",
    distill_teacher_hidden_dim: comparison.teacherHiddenDim ?? "n/a",
    distill_student_hidden_dim: comparison.studentHiddenDim ?? "n/a",
    distill_teacher_num_hidden_layers: comparison.teacherNumHiddenLayers ?? "n/a",
    distill_student_num_hidden_layers: comparison.studentNumHiddenLayers ?? "n/a",
    distill_teacher_input_dim: comparison.teacherInputDim ?? "n/a",
    distill_student_input_dim: comparison.studentInputDim ?? "n/a",
    distill_teacher_output_dim: comparison.teacherOutputDim ?? "n/a",
    distill_student_output_dim: comparison.studentOutputDim ?? "n/a",
    distill_teacher_model_size_bytes: comparison.teacherModelSizeBytes ?? "n/a",
    distill_student_model_size_bytes: comparison.studentModelSizeBytes ?? "n/a",
    distill_size_saved_bytes: comparison.sizeSavedBytes ?? "n/a",
    distill_size_saved_percent: comparison.sizeSavedPercent ?? "n/a",
    distill_teacher_param_count: comparison.teacherParamCount ?? "n/a",
    distill_student_param_count: comparison.studentParamCount ?? "n/a",
    distill_param_saved_count: comparison.paramSavedCount ?? "n/a",
    distill_param_saved_percent: comparison.paramSavedPercent ?? "n/a",
  };
}

/**
 * Resolves distillation modal payload from in-memory snapshot first, then persisted fallback rows.
 * @param params - Required parameters.
 * @returns Resolved payload or missing-state marker.
 */
export function resolveDistilledModalPayload(
  {
    teacherKey,
    snapshotsByTeacher,
    trainingRuns,
  }: {
    teacherKey: string;
    snapshotsByTeacher: Record<string, DistilledSnapshot>;
    trainingRuns: TrainingRunRow[];
  },
  {}: Record<string, never> = {}
):
  | {
      status: "snapshot";
      metrics: TrainingMetrics;
      modelId: string | null;
      modelPath: string | null;
      comparison: DistillComparison;
    }
  | {
      status: "fallback";
      metrics: TrainingMetrics;
      modelId: string;
      modelPath: string;
      comparison: DistillComparison;
    }
  | { status: "missing" } {
  const snapshot = snapshotsByTeacher[teacherKey];
  if (snapshot) {
    return {
      status: "snapshot",
      metrics: snapshot.metrics,
      modelId: snapshot.modelId,
      modelPath: snapshot.modelPath,
      comparison: snapshot.comparison,
    };
  }

  const fallbackDistilled = trainingRuns.find(
    (candidate) =>
      String(candidate.result ?? "") === "distilled" &&
      String(candidate.teacher_ref_key ?? "") === teacherKey
  );
  if (!fallbackDistilled) {
    return { status: "missing" };
  }

  const comparison: DistillComparison = {
    metricName: String(
      fallbackDistilled.distill_teacher_metric_name ??
        fallbackDistilled.metric_name ??
        "accuracy"
    ),
    teacherMetricValue: parseNumericValue({
      value: fallbackDistilled.distill_teacher_metric_value,
    }),
    studentMetricValue: parseNumericValue({
      value:
        fallbackDistilled.distill_student_metric_value ??
        fallbackDistilled.metric_score,
    }),
    qualityDelta: parseNumericValue({
      value: fallbackDistilled.distill_quality_delta,
    }),
    higherIsBetter: String(fallbackDistilled.distill_higher_is_better ?? "1") === "1",
    teacherTrainingMode: String(
      fallbackDistilled.distill_teacher_training_mode ?? "n/a"
    ),
    studentTrainingMode: String(
      fallbackDistilled.distill_student_training_mode ?? "n/a"
    ),
    teacherHiddenDim: parseNumericValue({
      value: fallbackDistilled.distill_teacher_hidden_dim,
    }),
    studentHiddenDim: parseNumericValue({
      value: fallbackDistilled.distill_student_hidden_dim,
    }),
    teacherNumHiddenLayers: parseNumericValue({
      value: fallbackDistilled.distill_teacher_num_hidden_layers,
    }),
    studentNumHiddenLayers: parseNumericValue({
      value: fallbackDistilled.distill_student_num_hidden_layers,
    }),
    teacherInputDim: parseNumericValue({
      value: fallbackDistilled.distill_teacher_input_dim,
    }),
    studentInputDim: parseNumericValue({
      value: fallbackDistilled.distill_student_input_dim,
    }),
    teacherOutputDim: parseNumericValue({
      value: fallbackDistilled.distill_teacher_output_dim,
    }),
    studentOutputDim: parseNumericValue({
      value: fallbackDistilled.distill_student_output_dim,
    }),
    teacherModelSizeBytes: parseNumericValue({
      value: fallbackDistilled.distill_teacher_model_size_bytes,
    }),
    studentModelSizeBytes: parseNumericValue({
      value: fallbackDistilled.distill_student_model_size_bytes,
    }),
    sizeSavedBytes: parseNumericValue({
      value: fallbackDistilled.distill_size_saved_bytes,
    }),
    sizeSavedPercent: parseNumericValue({
      value: fallbackDistilled.distill_size_saved_percent,
    }),
    teacherParamCount: parseNumericValue({
      value: fallbackDistilled.distill_teacher_param_count,
    }),
    studentParamCount: parseNumericValue({
      value: fallbackDistilled.distill_student_param_count,
    }),
    paramSavedCount: parseNumericValue({
      value: fallbackDistilled.distill_param_saved_count,
    }),
    paramSavedPercent: parseNumericValue({
      value: fallbackDistilled.distill_param_saved_percent,
    }),
  };

  return {
    status: "fallback",
    metrics: {
      task: String(fallbackDistilled.task ?? "auto"),
      train_loss: parseNumericValue({ value: fallbackDistilled.train_loss }) ?? undefined,
      test_loss: parseNumericValue({ value: fallbackDistilled.test_loss }) ?? undefined,
      test_metric_name: String(
        fallbackDistilled.metric_name ?? comparison.metricName
      ),
      test_metric_value:
        parseNumericValue({ value: fallbackDistilled.metric_score }) ?? undefined,
    },
    modelId: String(fallbackDistilled.model_id ?? "n/a"),
    modelPath: String(fallbackDistilled.model_path ?? "n/a"),
    comparison,
  };
}
