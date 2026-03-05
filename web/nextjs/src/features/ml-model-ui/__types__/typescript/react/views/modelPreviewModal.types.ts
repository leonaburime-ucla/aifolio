import type { Edge, Node } from "@xyflow/react";

export type ModelPreviewFramework = "pytorch" | "tensorflow";

export type ModelPreviewMode =
  | "mlp_dense"
  | "linear_glm_baseline"
  | "tabresnet"
  | "wide_and_deep"
  | "imbalance_aware"
  | "quantile_regression"
  | "calibrated_classifier"
  | "entity_embeddings"
  | "autoencoder_head"
  | "multi_task_learning"
  | "time_aware_tabular"
  | "tree_teacher_distillation";

export type ModelPreviewModalProps = {
  isOpen: boolean;
  onClose: () => void;
  framework: ModelPreviewFramework;
  mode: ModelPreviewMode;
};

export type GraphModel = {
  title: string;
  summary: string;
  nodes: Node[];
  edges: Edge[];
};

export type ModelPreviewTerminology = {
  term: string;
  definition: string;
};

export type ModelPreviewBullets = {
  layers: string[];
  terminology: ModelPreviewTerminology[];
};

export type ModelPreviewModel = {
  graph: GraphModel;
  data: ModelPreviewBullets;
};
