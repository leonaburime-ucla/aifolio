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
