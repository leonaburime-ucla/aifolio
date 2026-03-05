import { describe, expect, it } from "vitest";
import {
  buildDistillationComparison,
  buildEnrichedDistilledRun,
  resolveDistilledModalPayload,
} from "@/features/ml/typescript/logic/distillationView.logic";

describe("distillationView.logic", () => {
  it("builds comparison and quality delta for higher-is-better metrics", () => {
    const built = buildDistillationComparison({
      teacher: {
        metric_name: "accuracy",
        metric_score: "0.80",
        training_mode: "mlp_dense",
        hidden_dim: 128,
        num_hidden_layers: 3,
      },
      result: {
        metrics: { test_metric_name: "accuracy", test_metric_value: 0.9 },
        distilledRun: { training_mode: "mlp_dense", hidden_dim: 64, num_hidden_layers: 2 },
        teacherInputDim: 20,
        teacherOutputDim: 2,
        studentInputDim: 20,
        studentOutputDim: 2,
        teacherModelSizeBytes: 1000,
        studentModelSizeBytes: 500,
        sizeSavedBytes: 500,
        sizeSavedPercent: 50,
        teacherParamCount: 3000,
        studentParamCount: 1500,
        paramSavedCount: 1500,
        paramSavedPercent: 50,
      },
    });

    expect(built.teacherMetricName).toBe("accuracy");
    expect(built.qualityDelta).toBeCloseTo(0.1);
    expect(built.comparison.higherIsBetter).toBe(true);
  });

  it("builds comparison with null delta when student metric is missing", () => {
    const built = buildDistillationComparison({
      teacher: {
        metric_name: "loss",
        metric_score: "0.20",
      },
      result: {
        metrics: { test_metric_name: "loss" },
        distilledRun: {},
        teacherInputDim: null,
        teacherOutputDim: null,
        studentInputDim: null,
        studentOutputDim: null,
        teacherModelSizeBytes: null,
        studentModelSizeBytes: null,
        sizeSavedBytes: null,
        sizeSavedPercent: null,
        teacherParamCount: null,
        studentParamCount: null,
        paramSavedCount: null,
        paramSavedPercent: null,
      },
    });

    expect(built.studentMetricValue).toBeNull();
    expect(built.qualityDelta).toBeNull();
  });

  it("uses lower-is-better metric branch", () => {
    const built = buildDistillationComparison({
      teacher: {
        metric_name: "loss",
        metric_score: "0.25",
      },
      result: {
        metrics: { test_metric_name: "loss", test_metric_value: 0.2 },
        distilledRun: {},
        teacherInputDim: null,
        teacherOutputDim: null,
        studentInputDim: null,
        studentOutputDim: null,
        teacherModelSizeBytes: null,
        studentModelSizeBytes: null,
        sizeSavedBytes: null,
        sizeSavedPercent: null,
        teacherParamCount: null,
        studentParamCount: null,
        paramSavedCount: null,
        paramSavedPercent: null,
      },
    });

    expect(built.teacherMetricName).toBe("loss");
    expect(built.qualityDelta).toBeCloseTo(0.05);
  });

  it("enriches distilled run with teacher linkage fields", () => {
    const row = buildEnrichedDistilledRun({
      distilledRun: { run_id: "student-1", training_mode: "mlp_dense" },
      teacherKey: "teacher-1",
      comparison: {
        metricName: "accuracy",
        teacherMetricValue: 0.8,
        studentMetricValue: 0.9,
        qualityDelta: 0.1,
        higherIsBetter: true,
        teacherTrainingMode: "mlp_dense",
        studentTrainingMode: "mlp_dense",
        teacherHiddenDim: 128,
        studentHiddenDim: 64,
        teacherNumHiddenLayers: 3,
        studentNumHiddenLayers: 2,
        teacherInputDim: 20,
        studentInputDim: 20,
        teacherOutputDim: 2,
        studentOutputDim: 2,
        teacherModelSizeBytes: 1000,
        studentModelSizeBytes: 500,
        sizeSavedBytes: 500,
        sizeSavedPercent: 50,
        teacherParamCount: 3000,
        studentParamCount: 1500,
        paramSavedCount: 1500,
        paramSavedPercent: 50,
      },
      teacherMetricName: "accuracy",
      teacherMetricValue: 0.8,
      studentMetricValue: 0.9,
      qualityDelta: 0.1,
    });

    expect(row.teacher_ref_key).toBe("teacher-1");
    expect(row.distill_higher_is_better).toBe("1");
    expect(row.distill_teacher_hidden_dim).toBe(128);
  });

  it("writes n/a defaults for missing comparison metadata", () => {
    const row = buildEnrichedDistilledRun({
      distilledRun: { run_id: "student-2" },
      teacherKey: "teacher-2",
      comparison: {
        metricName: "accuracy",
        teacherMetricValue: null,
        studentMetricValue: null,
        qualityDelta: null,
        higherIsBetter: false,
        teacherTrainingMode: null,
        studentTrainingMode: null,
        teacherHiddenDim: null,
        studentHiddenDim: null,
        teacherNumHiddenLayers: null,
        studentNumHiddenLayers: null,
        teacherInputDim: null,
        studentInputDim: null,
        teacherOutputDim: null,
        studentOutputDim: null,
        teacherModelSizeBytes: null,
        studentModelSizeBytes: null,
        sizeSavedBytes: null,
        sizeSavedPercent: null,
        teacherParamCount: null,
        studentParamCount: null,
        paramSavedCount: null,
        paramSavedPercent: null,
      },
      teacherMetricName: "accuracy",
      teacherMetricValue: null,
      studentMetricValue: null,
      qualityDelta: null,
    });

    expect(row.distill_teacher_training_mode).toBe("n/a");
    expect(row.distill_student_training_mode).toBe("n/a");
    expect(row.distill_higher_is_better).toBe("0");
  });

  it("resolves snapshot, fallback, and missing payload states", () => {
    const snapshot = resolveDistilledModalPayload({
      teacherKey: "teacher-1",
      snapshotsByTeacher: {
        "teacher-1": {
          metrics: { test_metric_name: "accuracy", test_metric_value: 0.9 },
          modelId: "m1",
          modelPath: "/tmp/m1",
          comparison: {
            metricName: "accuracy",
            teacherMetricValue: 0.8,
            studentMetricValue: 0.9,
            qualityDelta: 0.1,
            higherIsBetter: true,
            teacherTrainingMode: "mlp_dense",
            studentTrainingMode: "mlp_dense",
            teacherHiddenDim: 128,
            studentHiddenDim: 64,
            teacherNumHiddenLayers: 3,
            studentNumHiddenLayers: 2,
            teacherInputDim: 20,
            studentInputDim: 20,
            teacherOutputDim: 2,
            studentOutputDim: 2,
            teacherModelSizeBytes: 1000,
            studentModelSizeBytes: 500,
            sizeSavedBytes: 500,
            sizeSavedPercent: 50,
            teacherParamCount: 3000,
            studentParamCount: 1500,
            paramSavedCount: 1500,
            paramSavedPercent: 50,
          },
        },
      },
      trainingRuns: [],
    });
    expect(snapshot.status).toBe("snapshot");

    const fallback = resolveDistilledModalPayload({
      teacherKey: "teacher-2",
      snapshotsByTeacher: {},
      trainingRuns: [
        {
          result: "completed",
          teacher_ref_key: "teacher-2",
        },
        {
          result: "distilled",
          teacher_ref_key: "teacher-2",
          model_id: "m2",
          model_path: "/tmp/m2",
          metric_name: undefined,
          metric_score: "0.86",
          distill_higher_is_better: "0",
          distill_teacher_training_mode: undefined,
          distill_student_training_mode: undefined,
        },
      ],
    });
    expect(fallback.status).toBe("fallback");
    if (fallback.status === "fallback") {
      expect(fallback.comparison.metricName).toBe("accuracy");
      expect(fallback.comparison.higherIsBetter).toBe(false);
      expect(fallback.metrics.test_metric_value).toBe(0.86);
    }

    const fallbackWithNonNumericMetric = resolveDistilledModalPayload({
      teacherKey: "teacher-3",
      snapshotsByTeacher: {},
      trainingRuns: [
        {
          result: "distilled",
          teacher_ref_key: undefined,
        },
        {
          result: "distilled",
          teacher_ref_key: "other-teacher",
        },
        {
          result: "distilled",
          teacher_ref_key: "teacher-3",
          model_id: undefined,
          model_path: undefined,
          metric_name: "loss",
          metric_score: "n/a",
        },
      ],
    });
    expect(fallbackWithNonNumericMetric.status).toBe("fallback");
    if (fallbackWithNonNumericMetric.status === "fallback") {
      expect(fallbackWithNonNumericMetric.metrics.test_metric_value).toBeUndefined();
      expect(fallbackWithNonNumericMetric.modelId).toBe("n/a");
      expect(fallbackWithNonNumericMetric.modelPath).toBe("n/a");
    }

    const missing = resolveDistilledModalPayload({
      teacherKey: "teacher-missing",
      snapshotsByTeacher: {},
      trainingRuns: [],
    });
    expect(missing).toEqual({ status: "missing" });
  });
});
