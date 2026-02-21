export const ADD_CHART_SPEC_TOOL = "add_chart_spec";
export const CLEAR_CHARTS_TOOL = "clear_charts";
export const NAVIGATE_TO_PAGE_TOOL = "navigate_to_page";
export const TRAIN_PYTORCH_MODEL_TOOL = "train_pytorch_model";

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
