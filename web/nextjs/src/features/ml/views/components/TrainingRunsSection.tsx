import DataTable from "@/core/views/components/Datatable/DataTable";
import {
  calcTrainingTableHeight,
  TRAINING_RUN_COLUMNS,
  type TrainingRunRow,
} from "@/features/ml/utils/trainingRuns.util";

type TrainingRunsSectionProps = {
  trainingRuns: TrainingRunRow[];
  copyRunsStatus: string | null;
  onCopyTrainingRuns: () => void;
  onClearTrainingRuns: () => void;
};

export function TrainingRunsSection({
  trainingRuns,
  copyRunsStatus,
  onCopyTrainingRuns,
  onClearTrainingRuns,
}: TrainingRunsSectionProps) {
  const trainingTableHeight = calcTrainingTableHeight(trainingRuns.length);

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
        </div>
      </div>
      {trainingRuns.length === 0 ? (
        <p className="text-xs text-zinc-500">
          No runs yet. Train once to populate the results table.
        </p>
      ) : (
        <DataTable
          rows={trainingRuns}
          columns={[...TRAINING_RUN_COLUMNS]}
          height={trainingTableHeight}
          maxWidth={980}
        />
      )}
    </div>
  );
}
