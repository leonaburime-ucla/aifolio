const ML_FORM_FIELD_ALIASES: Record<string, string> = {
  dataset: "dataset_id",
  datasetId: "dataset_id",
  epochs: "epoch_values",
  epoch: "epoch_values",
  epochValues: "epoch_values",
  batch_size: "batch_sizes",
  batchSize: "batch_sizes",
  batchSizes: "batch_sizes",
  learning_rate: "learning_rates",
  learningRate: "learning_rates",
  learningRates: "learning_rates",
  test_size: "test_sizes",
  testSize: "test_sizes",
  testSizes: "test_sizes",
  hidden_dim: "hidden_dims",
  hiddenDim: "hidden_dims",
  hiddenDims: "hidden_dims",
  num_hidden_layer: "num_hidden_layers",
  numHiddenLayer: "num_hidden_layers",
  numHiddenLayers: "num_hidden_layers",
  dropout: "dropouts",
  excludeColumns: "exclude_columns",
  dateColumns: "date_columns",
  model: "training_mode",
  model_type: "training_mode",
  modelType: "training_mode",
  algorithm: "training_mode",
  architecture: "training_mode",
  trainingMode: "training_mode",
  targetColumn: "target_column",
  runSweep: "run_sweep",
  autoDistill: "auto_distill",
  setSweepValues: "set_sweep_values",
};

const SUPPORTED_TRAINING_MODES = new Set([
  "mlp_dense",
  "linear_glm_baseline",
  "tabresnet",
  "imbalance_aware",
  "calibrated_classifier",
  "tree_teacher_distillation",
  "wide_and_deep",
  "quantile_regression",
  "entity_embeddings",
  "autoencoder_head",
  "multi_task_learning",
  "time_aware_tabular",
]);

const TRAINING_MODE_ALIASES: Record<string, string> = {
  neural_net: "mlp_dense",
  neural_network: "mlp_dense",
  neural_net_dense: "mlp_dense",
  dense_neural_net: "mlp_dense",
  dense_network: "mlp_dense",
  tab_resnet: "tabresnet",
  residual_mlp: "tabresnet",
  wide_deep: "wide_and_deep",
  wide_and_deep_model: "wide_and_deep",
};

const DATASET_ID_ALIASES: Record<string, string> = {
  customer_churn: "customer_churn_telco.csv",
  customer_churn_telco: "customer_churn_telco.csv",
  customer_churn_telco_csv: "customer_churn_telco.csv",
  churn: "customer_churn_telco.csv",
  fraud_detection: "fraud_detection_phishing_websites.csv",
  fraud: "fraud_detection_phishing_websites.csv",
  fraud_detection_phishing_websites: "fraud_detection_phishing_websites.csv",
  fraud_detection_phishing_websites_csv: "fraud_detection_phishing_websites.csv",
  house_prices: "house_prices_ames.csv",
  house_prices_ames: "house_prices_ames.csv",
  house_prices_ames_csv: "house_prices_ames.csv",
  sales_forecasting: "sales_forecasting_walmart.csv",
  sales_forecasting_walmart: "sales_forecasting_walmart.csv",
  sales_forecasting_walmart_csv: "sales_forecasting_walmart.csv",
  loan_default: "loan_default_credit_card_clients.xls",
  loan_default_credit_card_clients: "loan_default_credit_card_clients.xls",
  loan_default_credit_card_clients_xls: "loan_default_credit_card_clients.xls",
};

function normalizeLookupToken(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/&/g, "and")
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .replace(/_+/g, "_");
}

function normalizeTrainingModeValue(value: unknown): unknown {
  if (typeof value !== "string") return value;
  const token = normalizeLookupToken(value);
  if (TRAINING_MODE_ALIASES[token]) {
    return TRAINING_MODE_ALIASES[token];
  }
  if (SUPPORTED_TRAINING_MODES.has(token)) {
    return token;
  }
  return value;
}

function normalizeDatasetIdValue(value: unknown): unknown {
  if (typeof value !== "string") return value;
  const trimmed = value.trim();
  if (!trimmed) return value;
  const token = normalizeLookupToken(trimmed);
  if (DATASET_ID_ALIASES[token]) {
    return DATASET_ID_ALIASES[token];
  }
  return trimmed;
}

/**
 * Canonicalizes common model-emitted aliases into the ML form patch keys that
 * the bridge layer actually understands.
 *
 * This keeps Copilot tool calls resilient when the model chooses singular or
 * camelCase forms like `batch_size`, `testSize`, or `hidden_dim`.
 */
function normalizeMlFormPatchAliases(
  patch: Record<string, unknown>
): Record<string, unknown> {
  const normalized = { ...patch };
  for (const [alias, canonical] of Object.entries(ML_FORM_FIELD_ALIASES)) {
    if (normalized[canonical] !== undefined) continue;
    if (normalized[alias] === undefined) continue;
    normalized[canonical] = normalized[alias];
    delete normalized[alias];
  }
  if (normalized.training_mode !== undefined) {
    normalized.training_mode = normalizeTrainingModeValue(normalized.training_mode);
  }
  if (normalized.dataset_id !== undefined) {
    normalized.dataset_id = normalizeDatasetIdValue(normalized.dataset_id);
  }
  return normalized;
}

/**
 * Resolves common ML form patch args from Copilot tool payloads.
 *
 * Supports both payload styles:
 * - `{ fields: { ...patch } }`
 * - `{ ...patch }`
 */
export function resolveMlFormPatchFromToolArgs<TPatch extends Record<string, unknown>>(
  args: Record<string, unknown>
): TPatch {
  const patchCandidate = args.fields;
  const patch =
    patchCandidate && typeof patchCandidate === "object" && !Array.isArray(patchCandidate)
      ? { ...(patchCandidate as Record<string, unknown>) }
      : { ...args };
  const normalizedPatch = normalizeMlFormPatchAliases(patch);

  if (
    normalizedPatch.set_sweep_values !== undefined &&
    normalizedPatch.run_sweep === undefined
  ) {
    normalizedPatch.run_sweep = normalizedPatch.set_sweep_values;
  }

  return normalizedPatch as TPatch;
}
