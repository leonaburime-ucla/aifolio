import type {
  AddChartSpecHandler,
  AddChartSpecPayload,
  AgUiWorkspaceTab,
  NavigateToPageResult,
  SwitchAgUiTabResult,
} from "@aifolio/contracts/entities/ag-ui";
import { normalizeChartSpecInput } from "./copilotPayload";
import {
  isAllowedRoute,
  resolveRouteAlias,
  ROUTE_ALIASES,
} from "../config/frontendTools";
import { resolveAgUiWorkspaceTab } from "./workspace";

export function handleAddChartSpec(
  payload: AddChartSpecPayload,
  addChartSpec: AddChartSpecHandler
) {
  const combinedPayload = payload.chartSpecs ?? payload.chartSpec;
  const normalized = normalizeChartSpecInput(combinedPayload);

  if (!normalized) {
    return {
      status: "error" as const,
      code: "INVALID_CHART_SPEC" as const,
      addedCount: 0,
    };
  }

  const specs = Array.isArray(normalized) ? normalized : [normalized];
  specs.forEach((spec) => addChartSpec(spec));

  return {
    status: "ok" as const,
    addedCount: specs.length,
    ids: specs.map((spec) => spec.id),
  };
}

export function handleNavigateToPage(route: string): NavigateToPageResult {
  const resolvedRoute = resolveRouteAlias(route);
  if (!resolvedRoute || !isAllowedRoute(resolvedRoute)) {
    return {
      status: "error",
      code: "INVALID_ROUTE",
      allowedRoutes: Array.from(new Set(Object.values(ROUTE_ALIASES))),
    };
  }

  return {
    status: "ok",
    resolvedRoute,
  };
}

export function handleSwitchAgUiTab(tab: string): SwitchAgUiTabResult {
  const resolvedTab = resolveAgUiWorkspaceTab(tab);
  if (!resolvedTab) {
    return {
      status: "error",
      code: "INVALID_TAB",
      allowedTabs: ["charts", "agentic-research", "pytorch", "tensorflow"] as AgUiWorkspaceTab[],
    };
  }

  return {
    status: "ok",
    tab: resolvedTab,
  };
}
