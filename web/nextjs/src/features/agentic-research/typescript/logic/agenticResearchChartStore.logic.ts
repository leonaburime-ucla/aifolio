import type {
  AddChartSpecInput,
  AddChartSpecResult,
  ReorderChartSpecsInput,
  ReorderChartSpecsResult,
} from "@/features/agentic-research/__types__/typescript/logic/agenticResearchChartStore.types";

/**
 * Add chart spec by prepending newest and deduplicating by id.
 *
 * @param input - Required chart add inputs.
 * @returns Updated chart specs list.
 */
export function addChartSpecDedupPrepend(
  input: AddChartSpecInput
): AddChartSpecResult {
  return [input.spec, ...input.chartSpecs.filter((item) => item.id !== input.spec.id)];
}

/**
 * Reorder chart specs using ordered ids and append unspecified charts in current order.
 *
 * @param input - Required reorder inputs.
 * @returns Reordered chart specs list.
 */
export function reorderChartSpecsWithRemainder(
  input: ReorderChartSpecsInput
): ReorderChartSpecsResult {
  const byId = new Map(input.chartSpecs.map((spec) => [spec.id, spec]));
  const reordered = input.orderedIds
    .map((id) => byId.get(id))
    .filter((spec): spec is NonNullable<typeof spec> => Boolean(spec));
  const remaining = input.chartSpecs.filter(
    (spec) => !input.orderedIds.includes(spec.id)
  );
  return [...reordered, ...remaining];
}
