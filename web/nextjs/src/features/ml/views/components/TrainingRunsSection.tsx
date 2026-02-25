import DataTable from "@/core/views/components/Datatable/DataTable";
import {
  calcTrainingTableHeight,
  TRAINING_RUN_COLUMNS,
  type TrainingRunRow,
} from "@/features/ml/utils/trainingRuns.util";

type TrainingRunsSectionProps = {
  trainingRuns: TrainingRunRow[];
  copyRunsStatus: string | null;
  isTraining?: boolean;
  isStopRequested?: boolean;
  onCopyTrainingRuns: () => void;
  onClearTrainingRuns: () => void;
  onStopTrainingRuns?: () => void;
  onDistillFromRun?: (run: TrainingRunRow) => void;
  onSeeDistilledFromRun?: (run: TrainingRunRow) => void;
  distillingTeacherKey?: string | null;
  distilledByTeacher?: Record<string, string>;
};

export function TrainingRunsSection({
  trainingRuns,
  copyRunsStatus,
  isTraining = false,
  isStopRequested = false,
  onCopyTrainingRuns,
  onClearTrainingRuns,
  onStopTrainingRuns,
  onDistillFromRun,
  onSeeDistilledFromRun,
  distillingTeacherKey = null,
  distilledByTeacher = {},
}: TrainingRunsSectionProps) {
  const trainingTableHeight = calcTrainingTableHeight(trainingRuns.length);
  const cellRenderers = {
    distill_action: (_value: unknown, row: TrainingRunRow) => {
      const rowResult = String(row.result ?? "");
      if (rowResult === "distilled") {
        return (
          <span className="inline-flex rounded-md border border-emerald-200 bg-emerald-50 px-2 py-1 text-xs font-medium text-emerald-700">
            Student Model
          </span>
        );
      }

      const runId = String(row.run_id ?? "");
      const modelId = String(row.model_id ?? "");
      const modelPath = String(row.model_path ?? "");
      const teacherKey =
        (runId && runId !== "n/a" ? runId : "") ||
        (modelId && modelId !== "n/a" ? modelId : "") ||
        (modelPath && modelPath !== "n/a" ? modelPath : "");
      const isEligibleTeacher = Boolean(teacherKey) && rowResult === "completed";
      const isDistillingThisRow = distillingTeacherKey === teacherKey;
      const hasDistilled = Boolean(distilledByTeacher[teacherKey]);

      if (!isEligibleTeacher) {
        return <span className="text-xs text-zinc-400">n/a</span>;
      }

      if (hasDistilled) {
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
          disabled={!onDistillFromRun || isDistillingThisRow}
        >
          {isDistillingThisRow ? "Distilling..." : "Distill"}
        </button>
      );
    },
  } as const;

  return (
    <div className="mt-4 border-t border-zinc-200 pt-4">
      <div className="mb-2 flex items-center justify-between">
        <p className="text-xs font-semibold uppercase tracking-wide text-zinc-500">
          Training Runs
        </p>
        <div className="flex items-center gap-2">
          {copyRunsStatus ? (
            <span className="text-xs text-zinc-500">{copyRunsStatus}</span>
          ) : null}
          <button
            type="button"
            className="rounded-md border border-zinc-300 bg-white px-2 py-1 text-xs font-medium text-zinc-700 disabled:cursor-not-allowed disabled:text-zinc-400"
            onClick={onCopyTrainingRuns}
            disabled={trainingRuns.length === 0}
          >
            Copy Results
          </button>
          <button
            type="button"
            className="rounded-md bg-zinc-900 px-2 py-1 text-xs font-medium text-white disabled:cursor-not-allowed disabled:bg-zinc-400"
            onClick={onClearTrainingRuns}
            disabled={trainingRuns.length === 0}
          >
            Clear Runs
          </button>
          <button
            type="button"
            className="rounded-md bg-red-600 px-2 py-1 text-xs font-medium text-white transition-opacity disabled:cursor-not-allowed disabled:bg-red-300"
            onClick={onStopTrainingRuns}
            disabled={!isTraining || isStopRequested}
            aria-busy={isStopRequested}
          >
            {isStopRequested ? "Stop Requested..." : "Stop Training Runs"}
          </button>
        </div>
      </div>
      {isStopRequested ? (
        <p className="mb-2 text-xs text-amber-700">
          Stop requested. Current run will finish, then remaining runs are canceled.
        </p>
      ) : null}
      {trainingRuns.length === 0 ? (
        <p className="text-xs text-zinc-500">
          No runs yet. Train once to populate the results table.
        </p>
      ) : (
        <DataTable
          rows={trainingRuns}
          columns={[...TRAINING_RUN_COLUMNS]}
          cellRenderers={cellRenderers}
          height={trainingTableHeight}
          maxWidth={980}
        />
      )}
    </div>
  );
}
