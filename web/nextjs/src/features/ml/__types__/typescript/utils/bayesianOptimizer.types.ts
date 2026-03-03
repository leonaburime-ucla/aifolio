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
