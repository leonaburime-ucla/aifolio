import type {
  AddChartSpecHandler,
  AddChartSpecPayload,
  NavigateToPageResult,
  PytorchFormPatch,
  SwitchAgUiTabResult,
  TensorflowFormPatch,
} from "@/features/ag-ui-chat/__types__/typescript/logic/frontendTools.types";
import { normalizeChartSpecInput } from "@/features/ag-ui-chat/typescript/logic/copilotAssistantPayload.util";
import {
  isAllowedRoute,
  resolveRouteAlias,
  ROUTE_ALIASES,
} from "@/features/ag-ui-chat/typescript/config/frontendTools.config";
import { resolveAgUiWorkspaceTab } from "@/features/ag-ui-chat/typescript/logic/agUiWorkspace.logic";
import { resolveMlFormPatchFromToolArgs } from "@/features/ml/typescript/ai/agUi/mlTrainingTools.logic";

/**
 * Pure AG-UI frontend tool logic.
 *
 * Purpose:
 * - Normalize and validate tool-call payloads.
 * - Resolve route/tab aliases and return typed outcomes.
 * - Keep deterministic logic free of browser globals and network I/O.
 *
 * Layering:
 * - This file is intentionally pure (Business Logic style).
 * - Browser/API side effects live in `frontendTools.adapter.ts`.
 */

/**
 * Normalizes incoming chart payload(s) and dispatches valid chart specs.
 *
 * @param payload Tool-call payload containing a single `chartSpec` or array `chartSpecs`.
 * @param addChartSpec Callback that writes a normalized chart spec into the chart store.
 * @returns Status object including count and generated spec ids when successful.
 */
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

/**
 * Resolves and validates navigation routes for the `navigate_to_page` tool.
 *
 * @param route Raw route alias or path from the tool call.
 * @returns A validated route result or an error with allowed routes.
 */
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

/**
 * Resolves/validates an AG-UI tab token for the `switch_ag_ui_tab` tool.
 *
 * @param tab Raw tab alias from the tool call.
 * @returns Resolved tab on success or an error with allowed tabs.
 */
export function handleSwitchAgUiTab(tab: string): SwitchAgUiTabResult {
  const resolvedTab = resolveAgUiWorkspaceTab(tab);
  if (!resolvedTab) {
    return {
      status: "error",
      code: "INVALID_TAB",
      allowedTabs: ["charts", "agentic-research", "pytorch", "tensorflow"],
    };
  }

  return {
    status: "ok",
    tab: resolvedTab,
  };
}

/**
 * Resolves `set_pytorch_form_fields` tool args into a normalized patch object.
 *
 * Supports both payload styles:
 * - `{ fields: { ...patch } }`
 * - `{ ...patch }`
 *
 * @param args Raw tool args object.
 * @returns Normalized PyTorch form patch object.
 */
export function resolvePytorchFormPatchFromToolArgs(
  args: Record<string, unknown>
): PytorchFormPatch {
  return resolveMlFormPatchFromToolArgs<PytorchFormPatch>(args);
}

/**
 * Resolves `set_tensorflow_form_fields` tool args into a normalized patch object.
 *
 * Supports both payload styles:
 * - `{ fields: { ...patch } }`
 * - `{ ...patch }`
 *
 * @param args Raw tool args object.
 * @returns Normalized TensorFlow form patch object.
 */
export function resolveTensorflowFormPatchFromToolArgs(
  args: Record<string, unknown>
): TensorflowFormPatch {
  return resolveMlFormPatchFromToolArgs<TensorflowFormPatch>(args);
}
