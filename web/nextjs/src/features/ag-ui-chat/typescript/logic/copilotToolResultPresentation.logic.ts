import type {
  NavigateToPageResult,
  SwitchAgUiTabResult,
} from "@/features/ag-ui-chat/__types__/typescript/logic/frontendTools.types";

type StatusLikeResult = {
  status?: string;
  code?: string;
  applied?: string[];
  addedCount?: number;
  resolvedRoute?: string;
  run_id?: string;
  tab?: string;
};

function humanizeFieldName(field: string): string {
  return field.replace(/_/g, " ");
}

function formatFieldList(fields: string[]): string {
  return fields.map(humanizeFieldName).join(", ");
}

function getErrorCode(result: unknown): string {
  if (!result || typeof result !== "object") return "UNKNOWN_ERROR";
  const code = (result as { code?: unknown }).code;
  return typeof code === "string" && code.trim().length > 0 ? code : "UNKNOWN_ERROR";
}

/**
 * Formats add-chart tool results into concise transcript-safe text.
 *
 * Returning prose instead of raw objects prevents Copilot from dumping JSON
 * tool payloads into the visible assistant transcript.
 */
export function formatAddChartSpecToolResult(result: StatusLikeResult): string {
  if (result.status === "ok") {
    const count = typeof result.addedCount === "number" ? result.addedCount : 0;
    return count === 1 ? "Added 1 chart." : `Added ${count} charts.`;
  }
  return `Unable to add chart: ${getErrorCode(result)}.`;
}

/**
 * Formats clear-chart tool results into concise transcript-safe text.
 */
export function formatClearChartsToolResult(): string {
  return "Cleared charts.";
}

/**
 * Formats navigation tool results into concise transcript-safe text.
 */
export function formatNavigateToPageToolResult(result: NavigateToPageResult): string {
  if (result.status === "ok") {
    return `Navigated to ${result.resolvedRoute}.`;
  }
  return `Unable to navigate: ${result.code}.`;
}

/**
 * Formats AG-UI tab-switch results into concise transcript-safe text.
 */
export function formatSwitchAgUiTabToolResult(result: SwitchAgUiTabResult): string {
  if (result.status === "ok") {
    return `Switched to the ${result.tab} tab.`;
  }
  return `Unable to switch tabs: ${result.code}.`;
}

/**
 * Formats form-patch tool results into concise transcript-safe text.
 */
export function formatSetFormFieldsToolResult(frameworkLabel: string, result: StatusLikeResult): string {
  if (result.status === "ok") {
    const applied = Array.isArray(result.applied) ? result.applied : [];
    return applied.length > 0
      ? `Updated ${frameworkLabel} fields: ${formatFieldList(applied)}.`
      : `Updated ${frameworkLabel} form fields.`;
  }
  return `Unable to update ${frameworkLabel} form fields: ${getErrorCode(result)}.`;
}

/**
 * Formats target-column tool results into concise transcript-safe text.
 */
export function formatChangeTargetColumnToolResult(
  frameworkLabel: string,
  targetColumn: string | undefined,
  result: StatusLikeResult
): string {
  if (result.status === "ok") {
    return targetColumn && targetColumn.trim().length > 0
      ? `Changed ${frameworkLabel} target column to ${targetColumn.trim()}.`
      : `Changed ${frameworkLabel} target column.`;
  }
  return `Unable to change ${frameworkLabel} target column: ${getErrorCode(result)}.`;
}

/**
 * Formats randomize tool results into concise transcript-safe text.
 */
export function formatRandomizeFormFieldsToolResult(
  frameworkLabel: string,
  result: StatusLikeResult
): string {
  if (result.status === "ok") {
    return `Randomized ${frameworkLabel} form fields.`;
  }
  return `Unable to randomize ${frameworkLabel} form fields: ${getErrorCode(result)}.`;
}

/**
 * Formats "start training runs" tool results into concise transcript-safe text.
 */
export function formatStartTrainingRunsToolResult(
  frameworkLabel: string,
  result: StatusLikeResult
): string {
  if (result.status === "ok") {
    return `Started ${frameworkLabel} training runs.`;
  }
  return `Unable to start ${frameworkLabel} training runs: ${getErrorCode(result)}.`;
}

/**
 * Formats single-run training tool results into concise transcript-safe text.
 */
export function formatTrainModelToolResult(frameworkLabel: string, result: StatusLikeResult): string {
  if (result.status === "ok") {
    const runId =
      typeof result.run_id === "string" && result.run_id.trim().length > 0
        ? ` (${result.run_id.trim()})`
        : "";
    return `Started one ${frameworkLabel} training run${runId}.`;
  }
  return `Unable to start ${frameworkLabel} training: ${getErrorCode(result)}.`;
}
