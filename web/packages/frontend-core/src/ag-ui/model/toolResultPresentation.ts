import type { NavigateToPageResult, SwitchAgUiTabResult } from "@aifolio/contracts/entities/ag-ui";

type StatusLikeResult = {
  status?: string;
  code?: string;
  applied?: string[];
  addedCount?: number;
  resolvedRoute?: string;
  run_id?: string;
  tab?: string;
};

export function humanizeFieldName(field: string): string {
  return field.replace(/_/g, " ");
}

export function formatFieldList(fields: string[]): string {
  return fields.map(humanizeFieldName).join(", ");
}

export function getErrorCode(result: unknown): string {
  if (!result || typeof result !== "object") return "UNKNOWN_ERROR";
  const code = (result as { code?: unknown }).code;
  return typeof code === "string" && code.trim().length > 0 ? code : "UNKNOWN_ERROR";
}

export function formatAddChartSpecToolResult(result: StatusLikeResult): string {
  if (result.status === "ok") {
    const count = typeof result.addedCount === "number" ? result.addedCount : 0;
    return count === 1 ? "Added 1 chart." : `Added ${count} charts.`;
  }
  return `Unable to add chart: ${getErrorCode(result)}.`;
}

export function formatClearChartsToolResult(): string {
  return "Cleared charts.";
}

export function formatNavigateToPageToolResult(result: NavigateToPageResult): string {
  if (result.status === "ok") {
    return `Navigated to ${result.resolvedRoute}.`;
  }
  return `Unable to navigate: ${result.code}.`;
}

export function formatSwitchAgUiTabToolResult(result: SwitchAgUiTabResult): string {
  if (result.status === "ok") {
    return `Switched to the ${result.tab} tab.`;
  }
  return `Unable to switch tabs: ${result.code}.`;
}

export function formatSetFormFieldsToolResult(frameworkLabel: string, result: StatusLikeResult): string {
  if (result.status === "ok") {
    const applied = Array.isArray(result.applied) ? result.applied : [];
    return applied.length > 0
      ? `Updated ${frameworkLabel} fields: ${formatFieldList(applied)}.`
      : `Updated ${frameworkLabel} form fields.`;
  }
  return `Unable to update ${frameworkLabel} form fields: ${getErrorCode(result)}.`;
}

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

export function formatRandomizeFormFieldsToolResult(
  frameworkLabel: string,
  result: StatusLikeResult
): string {
  if (result.status === "ok") {
    return `Randomized ${frameworkLabel} form fields.`;
  }
  return `Unable to randomize ${frameworkLabel} form fields: ${getErrorCode(result)}.`;
}

export function formatStartTrainingRunsToolResult(
  frameworkLabel: string,
  result: StatusLikeResult
): string {
  if (result.status === "ok") {
    return `Started ${frameworkLabel} training runs.`;
  }
  return `Unable to start ${frameworkLabel} training runs: ${getErrorCode(result)}.`;
}

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
