export const ADD_CHART_SPEC_TOOL = "add_chart_spec";
export const CLEAR_CHARTS_TOOL = "clear_charts";
export const NAVIGATE_TO_PAGE_TOOL = "navigate_to_page";
export const TRAIN_PYTORCH_MODEL_TOOL = "train_pytorch_model";
export const START_PYTORCH_TRAINING_RUNS_TOOL = "start_pytorch_training_runs";
export const TRAIN_TENSORFLOW_MODEL_TOOL = "train_tensorflow_model";
export const START_TENSORFLOW_TRAINING_RUNS_TOOL = "start_tensorflow_training_runs";
export const SWITCH_AG_UI_TAB_TOOL = "switch_ag_ui_tab";
export const SET_ACTIVE_ML_FORM_FIELDS_TOOL = "set_active_ml_form_fields";
export const CHANGE_ACTIVE_ML_TARGET_COLUMN_TOOL = "change_active_ml_target_column";
export const RANDOMIZE_ACTIVE_ML_FORM_FIELDS_TOOL = "randomize_active_ml_form_fields";
export const START_ACTIVE_ML_TRAINING_RUNS_TOOL = "start_active_ml_training_runs";
export const SET_PYTORCH_FORM_FIELDS_TOOL = "set_pytorch_form_fields";
export const CHANGE_PYTORCH_TARGET_COLUMN_TOOL = "change_pytorch_target_column";
export const RANDOMIZE_PYTORCH_FORM_FIELDS_TOOL = "randomize_pytorch_form_fields";
export const SET_TENSORFLOW_FORM_FIELDS_TOOL = "set_tensorflow_form_fields";
export const CHANGE_TENSORFLOW_TARGET_COLUMN_TOOL = "change_tensorflow_target_column";
export const RANDOMIZE_TENSORFLOW_FORM_FIELDS_TOOL = "randomize_tensorflow_form_fields";

export const ROUTE_ALIASES: Record<string, string> = {
  "/": "/",
  home: "/",
  "ai chat": "/",
  chat: "/",
  "ag-ui": "/ag-ui",
  agui: "/ag-ui",
  "agentic research": "/agentic-research",
  "agentic-research": "/agentic-research",
  pytorch: "/ml/pytorch",
  tensorflow: "/ml/tensorflow",
  "knowledge distillation": "/ml/knowledge-distillation",
  "knowledge-distillation": "/ml/knowledge-distillation",
};

export function resolveRouteAlias(route: string): string {
  const normalizedKey = String(route || "")
    .trim()
    .toLowerCase();
  return ROUTE_ALIASES[normalizedKey] || (route?.startsWith("/") ? route : "");
}

export function isAllowedRoute(route: string): boolean {
  return Object.values(ROUTE_ALIASES).includes(route);
}
