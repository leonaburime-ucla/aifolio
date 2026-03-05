import type { IntegrationComposeArgs } from "@/features/ml/__types__/typescript/react/hooks/trainingIntegration.types";

/**
 * Shared integration hook utilities for ML training pages.
 *
 * Relative to other hooks in this folder:
 * - Framework hooks (`usePytorchTraining*`, `useTensorflowTraining*`) own framework behavior.
 * - This module owns cross-framework composition glue only.
 */

/**
 * Shared ML training integration composition hook.
 *
 * Why:
 * - Keeps framework hook files focused on framework-specific state + logic.
 * - Centralizes dataset/training-runs wiring in one place.
 * - Makes page-level integration behavior consistent and easier to test.
 *
 * @param args Integration composition dependencies.
 * @param args.useDatasetState Hook that creates dataset state for the current integration.
 * @param args.useUiState Hook that creates framework-specific UI state.
 * @param args.useLogic Hook that composes framework-specific logic from dataset/runs/ui.
 * @param args.useTrainingRunsState Hook that provides training-runs state/actions.
 * @returns Combined integration surface (dataset + ui + logic + run store controls).
 */
export function useTrainingIntegration<
  TDatasetState extends object,
  TUiState extends object,
  TLogic extends object,
>({
  useDatasetState,
  useUiState,
  useLogic,
  useTrainingRunsState,
}: IntegrationComposeArgs<TDatasetState, TUiState, TLogic>) {
  const dataset = useDatasetState();
  const { trainingRuns, prependTrainingRun, clearTrainingRuns } = useTrainingRunsState();
  const ui = useUiState();
  const logic = useLogic({
    dataset,
    trainingRuns,
    prependTrainingRun,
    ui,
  });

  return {
    ...dataset,
    ...ui,
    ...logic,
    trainingRuns,
    clearTrainingRuns,
  };
}
