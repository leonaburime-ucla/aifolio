import { normalizeChartSpecInput } from "../../ag-ui/model/copilotPayload";
import type { ChartSpec } from "@aifolio/contracts/entities/chart";
import type {
  ChartSpecSnapshot,
  ClearChartsResponse,
  RemoveChartSpecErrorResponse,
  RemoveChartSpecSuccessResponse,
  ReorderChartSpecsIndexErrorResponse,
  ReorderChartSpecsPayloadErrorResponse,
  ReorderChartSpecsSuccessResponse,
} from "@aifolio/contracts/entities/agentic-research";

export function handleAgenticAddChartSpec(
  payload: { chartSpec?: unknown; chartSpecs?: unknown[] },
  addChartSpec: (spec: ChartSpec) => void
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

export function handleAgenticClearCharts(clearFn: () => void): ClearChartsResponse {
  clearFn();
  return { status: "ok", cleared: true };
}

export function handleAgenticRemoveChartSpec(
  chartId: string,
  getSnapshot: () => ChartSpecSnapshot,
  removeFn: (id: string) => void
): RemoveChartSpecSuccessResponse | RemoveChartSpecErrorResponse {
  const current = getSnapshot().chartSpecs;
  const exists = current.some((spec) => spec.id === chartId);

  if (!exists) {
    return {
      status: "error",
      code: "CHART_NOT_FOUND",
      chart_id: chartId,
      available_chart_ids: current.map((spec) => spec.id),
    };
  }

  removeFn(chartId);

  return {
    status: "ok",
    removed_chart_id: chartId,
    remaining_count: getSnapshot().chartSpecs.length,
  };
}

export function handleAgenticReorderChartSpecs(
  args: {
    ordered_ids?: string[];
    from_index?: number;
    to_index?: number;
  },
  getSnapshot: () => ChartSpecSnapshot,
  reorderFn: (orderedIds: string[]) => void
):
  | ReorderChartSpecsSuccessResponse
  | ReorderChartSpecsIndexErrorResponse
  | ReorderChartSpecsPayloadErrorResponse {
  const { ordered_ids, from_index, to_index } = args;
  const current = getSnapshot().chartSpecs;

  if (Array.isArray(ordered_ids) && ordered_ids.length > 0) {
    reorderFn(ordered_ids);
    return {
      status: "ok",
      mode: "ordered_ids",
      chart_ids: getSnapshot().chartSpecs.map((spec) => spec.id),
    };
  }

  if (
    typeof from_index === "number" &&
    typeof to_index === "number" &&
    Number.isInteger(from_index) &&
    Number.isInteger(to_index)
  ) {
    if (
      from_index < 0 ||
      from_index >= current.length ||
      to_index < 0 ||
      to_index >= current.length
    ) {
      return {
        status: "error",
        code: "INDEX_OUT_OF_RANGE",
        from_index,
        to_index,
        chart_count: current.length,
      };
    }

    const ids = current.map((spec) => spec.id);
    const [moved] = ids.splice(from_index, 1);
    ids.splice(to_index, 0, moved);
    reorderFn(ids);

    return {
      status: "ok",
      mode: "index_move",
      chart_ids: getSnapshot().chartSpecs.map((spec) => spec.id),
    };
  }

  return {
    status: "error",
    code: "INVALID_REORDER_PAYLOAD",
    hint: "Provide ordered_ids or both from_index and to_index.",
  };
}
