import type { PytorchTrainingMode } from "@/features/ml/__types__/typescript/api/pytorchApi.types";
import type { TensorflowTrainingMode } from "@/features/ml/__types__/typescript/api/tensorflowApi.types";

type ModeExplainer = {
  what: string;
  why: string;
  distillationNote: string;
};

const FALLBACK_MODE_EXPLAINER: ModeExplainer = {
  what: "This training mode is available, but its detailed explainer copy is not registered yet.",
  why:
    "The UI falls back to a generic description so unsupported or newly added runtime modes do not crash the page.",
  distillationNote:
    "Distillation support depends on the current runtime mode and training hook guards.",
};

/**
 * Centralized mode explanations for ML training UIs.
 * Keeping these dictionaries in feature config avoids route-level business text duplication
 * and keeps pages focused on view composition.
 */
export const PYTORCH_MODE_EXPLAINERS: Record<PytorchTrainingMode, ModeExplainer> = {
  mlp_dense: {
    what:
      "Neural Net (dense network): learns nonlinear feature interactions using hidden layers with backpropagation.",
    why:
      "Flexible general-purpose model that captures complex interactions a linear baseline can miss, and works well as your default deep tabular learner.",
    distillationNote: "Supported.",
  },
  linear_glm_baseline: {
    what:
      "Linear/GLM baseline: a single linear head (logistic or linear regression) with no hidden stack.",
    why:
      "Fastest and easiest to interpret, making it ideal for benchmarking and sanity checks before moving to more complex models.",
    distillationNote: "Supported.",
  },
  tabresnet: {
    what:
      "TabResNet (Residual MLP): a dense network with residual skip connections between hidden blocks.",
    why:
      "Supports deeper tabular networks with better gradient flow and training stability than a plain dense stack.",
    distillationNote: "Supported.",
  },
  imbalance_aware: {
    what:
      "Imbalance-aware classifier: neural net training with class-weighted loss so minority classes matter more.",
    why:
      "Useful when target classes are skewed (for example 95/5 splits).",
    distillationNote:
      "Not supported yet because current distillation flow assumes standard unweighted teacher objectives.",
  },
  calibrated_classifier: {
    what:
      "Calibrated classifier (label-smoothed): classification training with label smoothing to reduce overconfident probabilities.",
    why:
      "Produces more conservative probability outputs that can improve confidence calibration in production decisions.",
    distillationNote:
      "Not supported yet because the current distillation objective does not preserve calibration-specific behavior.",
  },
  tree_teacher_distillation: {
    what:
      "Tree-teacher distillation: first trains a tree ensemble teacher, then trains a compact neural student to mimic the teacher and the target labels.",
    why:
      "Combines strong tabular tree patterns with a deployable neural student.",
    distillationNote:
      "This mode already includes teacher-student training, so separate post-run distillation is not supported.",
  },
};

/**
 * TensorFlow mode explainer copy shared by route-level UI cards.
 */
export const TENSORFLOW_MODE_EXPLAINERS: Record<TensorflowTrainingMode, ModeExplainer> = {
  mlp_dense: {
    what:
      "Neural Net (dense network): learns nonlinear feature interactions using hidden layers with backpropagation.",
    why:
      "Flexible general-purpose model that captures complex interactions a linear baseline can miss, and works well as your default deep tabular learner.",
    distillationNote: "Supported.",
  },
  linear_glm_baseline: {
    what:
      "Linear/GLM baseline: a single linear head (logistic or linear regression) with no hidden stack.",
    why:
      "Fastest and easiest to interpret, making it ideal for benchmarking and sanity checks before moving to more complex models.",
    distillationNote: "Supported.",
  },
  wide_and_deep: {
    what:
      "Wide & Deep: combines a linear branch and deep branch, then merges both signals into one prediction.",
    why:
      "Balances memorization (wide) and generalization (deep), which is often stronger than either branch alone on mixed tabular feature sets.",
    distillationNote: "Supported.",
  },
  imbalance_aware: {
    what:
      "Imbalance-aware classifier: neural net training with class-weighted loss so minority classes matter more.",
    why:
      "Useful when target classes are highly skewed.",
    distillationNote:
      "Not supported yet because current distillation flow assumes standard unweighted teacher objectives.",
  },
  quantile_regression: {
    what:
      "Quantile regression: predicts a target quantile (for example P80) using pinball loss instead of mean-squared error.",
    why:
      "Adds uncertainty-aware forecasting by targeting distribution tails instead of only the mean.",
    distillationNote:
      "Not supported yet because current distillation focuses on point-estimate regression/classification targets.",
  },
  calibrated_classifier: {
    what:
      "Calibrated classifier (label-smoothed): classification training with label smoothing to reduce overconfident probabilities.",
    why:
      "Produces more conservative probability outputs that can improve confidence calibration in production decisions.",
    distillationNote:
      "Not supported yet because the current distillation objective does not preserve calibration-specific behavior.",
  },
  entity_embeddings: {
    what:
      "Entity embeddings model: learns dense latent feature representations before prediction instead of relying only on raw one-hot patterns.",
    why:
      "Can compress sparse categorical structure into compact learned factors that improve generalization on high-cardinality tabular data.",
    distillationNote:
      "Not supported yet because the distillation path currently assumes the standard feature-to-head student architecture, not embedding-projection-specific teacher behavior.",
  },
  autoencoder_head: {
    what:
      "Autoencoder + head: learns a compressed latent representation and jointly reconstructs inputs while predicting the target.",
    why:
      "Adds a representation-learning objective that can improve robustness and signal extraction from noisy tabular features.",
    distillationNote:
      "Not supported yet because this is a multi-output training objective (prediction + reconstruction), while distillation currently expects a single supervised output path.",
  },
  multi_task_learning: {
    what:
      "Multi-task learning: shared trunk with a primary prediction head plus an auxiliary head trained jointly.",
    why:
      "Shared representation from related objectives can improve generalization compared with single-task training.",
    distillationNote:
      "Not supported yet because the current distillation objective is single-head, and this mode trains main + auxiliary heads jointly.",
  },
  time_aware_tabular: {
    what:
      "Time-aware tabular model: applies a temporal gating path over date-expanded features before deep prediction layers.",
    why:
      "Helps the model focus on temporal structure in ordered/date-derived features beyond plain static tabular treatment.",
    distillationNote:
      "Not supported yet because this mode adds temporal gating branches that are not represented in the current distillation student template.",
  },
};

export function getPytorchModeExplainer(mode: string): ModeExplainer {
  return PYTORCH_MODE_EXPLAINERS[mode as PytorchTrainingMode] ?? FALLBACK_MODE_EXPLAINER;
}

export function getTensorflowModeExplainer(mode: string): ModeExplainer {
  return TENSORFLOW_MODE_EXPLAINERS[mode as TensorflowTrainingMode] ?? FALLBACK_MODE_EXPLAINER;
}
