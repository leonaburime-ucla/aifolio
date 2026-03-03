import { useMemo } from "react";
import { calcTrainingTableHeight } from "@/features/ml/typescript/utils/trainingRuns.util";
import { buildDistillActionModel } from "@/features/ml/typescript/react/logic/trainingRunsSection.logic";
import type { TrainingRunRow } from "@/features/ml/__types__/typescript/utils/trainingRuns.types";
import type {
  UseTrainingRunsSectionModelParams,
  UseTrainingRunsSectionModelResult,
} from "@/features/ml/__types__/typescript/react/hooks/trainingRunsSection.hooks.types";

type TrainingRunsCellRenderers = {
  distill_action: (_value: unknown, row: TrainingRunRow) => JSX.Element;
};

/**
 * Builds derived layout and cell-render behavior for the training runs table.
 * @param params - Required parameters.
 * @returns View-model values for `TrainingRunsSection`.
 */
export function useTrainingRunsSectionModel({
  trainingRuns,
  onDistillFromRun,
  onSeeDistilledFromRun,
  isDistillationSupportedForRun,
  distillingTeacherKey = null,
  distilledByTeacher = {},
}: UseTrainingRunsSectionModelParams): UseTrainingRunsSectionModelResult {
  const trainingTableHeight = useMemo(
    () => calcTrainingTableHeight({ rowsCount: trainingRuns.length }),
    [trainingRuns.length]
  );

  const cellRenderers = useMemo<TrainingRunsCellRenderers>(() => {
    return {
      distill_action: (_value: unknown, row: TrainingRunRow) => {
        const action = buildDistillActionModel({
          row,
          isDistillationSupportedForRun,
          distillingTeacherKey,
          distilledByTeacher,
        });

        if (action.kind === "student_model") {
          return (
            <span className="inline-flex rounded-md border border-emerald-200 bg-emerald-50 px-2 py-1 text-xs font-medium text-emerald-700">
              Student Model
            </span>
          );
        }

        if (action.kind === "not_available") {
          return <span className="text-xs text-zinc-400">Not Available</span>;
        }

        if (action.kind === "show_distilled") {
          return (
            <button
              type="button"
              className="rounded-md border border-zinc-300 bg-white px-2 py-1 text-xs font-medium text-zinc-700"
              onClick={() => onSeeDistilledFromRun?.(row)}
            >
              Show Distilled
            </button>
          );
        }

        return (
          <button
            type="button"
            className="rounded-md bg-zinc-900 px-2 py-1 text-xs font-medium text-white disabled:cursor-not-allowed disabled:bg-zinc-400"
            onClick={() => onDistillFromRun?.(row)}
            disabled={!onDistillFromRun || action.isDistillingThisRow}
          >
            {action.isDistillingThisRow ? "Distilling..." : "Distill"}
          </button>
        );
      },
    };
  }, [
    distillingTeacherKey,
    distilledByTeacher,
    isDistillationSupportedForRun,
    onDistillFromRun,
    onSeeDistilledFromRun,
  ]);

  return { trainingTableHeight, cellRenderers };
}
