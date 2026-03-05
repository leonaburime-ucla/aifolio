import type { TensorflowTrainingMode } from "@/features/ml/__types__/typescript/api/tensorflowApi.types";
import { useTrainingIntegration } from "@/features/ml/typescript/react/hooks/training.hooks";
import {
  useTrainingFrameworkLogic,
  useTrainingFrameworkUiState,
} from "@/features/ml/typescript/react/hooks/trainingFramework.hooks";
import type {
  TensorflowIntegrationDeps,
  TensorflowIntegrationArgs,
  TensorflowLogicDeps,
  TensorflowLogicArgs,
  TensorflowUiStateDeps,
} from "@/features/ml/__types__/typescript/react/hooks/tensorflowTraining.types";
export type {
  TensorflowIntegrationArgs,
  TensorflowLogicArgs,
  TensorflowRuntimeDeps,
} from "@/features/ml/__types__/typescript/react/hooks/tensorflowTraining.types";

export type { TensorflowTrainingMode };

/**
 * TensorFlow-specific training hooks module.
 *
 * Relative to other hooks in this folder:
 * - `trainingFramework.hooks.ts` owns shared framework-agnostic training behavior.
 * - This file only supplies TensorFlow adapters and exported wrapper hook names.
 */

const TENSORFLOW_DISTILL_SUPPORTED_MODES: TensorflowTrainingMode[] = [
  "mlp_dense",
  "linear_glm_baseline",
  "wide_and_deep",
];

/**
 * Type-guard for distillation-supported TensorFlow modes.
 *
 * @param mode Raw training mode token.
 * @returns True when the provided mode supports current distillation flow.
 */
function isTensorflowDistillSupportedMode(mode: string): mode is TensorflowTrainingMode {
  return TENSORFLOW_DISTILL_SUPPORTED_MODES.includes(mode as TensorflowTrainingMode);
}

/**
 * Creates TensorFlow UI state by extending shared ML training UI state with
 * framework-specific `trainingMode`.
 *
 * @returns TensorFlow UI state slice with training mode controls.
 */
export function useTensorflowUiState() {
  return useTensorflowUiStateWithDeps();
}

/**
 * Creates TensorFlow UI state with injectable shared-hook dependency.
 *
 * @param deps Dependency overrides for UI state composition.
 * @param deps.useFrameworkUiState Shared framework UI-state hook.
 * @returns TensorFlow UI state slice with training mode controls.
 */
export function useTensorflowUiStateWithDeps(
  deps: Partial<TensorflowUiStateDeps> = {}
) {
  const { useFrameworkUiState } = { useFrameworkUiState: useTrainingFrameworkUiState, ...deps };
  return useFrameworkUiState<TensorflowTrainingMode>("wide_and_deep");
}

/**
 * Core TensorFlow training workflow logic.
 * Handles dataset defaults, sweep planning, training execution, and distillation flows.
 *
 * @param args Logic dependencies (dataset state, run state, ui state, orchestrators, runtime deps).
 * @param deps Optional dependency overrides for shared framework logic composition.
 * @returns TensorFlow training behavior API consumed by integration/page layers.
 */
export function useTensorflowLogic(
  args: TensorflowLogicArgs,
  deps: Partial<TensorflowLogicDeps> = {}
) {
  const { useFrameworkLogic } = { useFrameworkLogic: useTrainingFrameworkLogic, ...deps };
  return useFrameworkLogic({
    ...args,
    framework: {
      isDistillationSupportedMode: isTensorflowDistillSupportedMode,
      trainModel: args.trainModel,
      distillModel: args.distillModel,
    },
  });
}

/**
 * Public integration hook consumed by pages/components.
 * Wires dataset state, run-store state, UI state, and TensorFlow logic into one surface.
 *
 * @param args Injected dataset/runs/model/training orchestration dependencies.
 * @param deps Optional dependency overrides for integration composition.
 * @returns Unified TensorFlow training integration model for UI composition.
 */
export function useTensorflowTrainingIntegration({
  useDatasetState,
  useTrainingRunsState,
  trainModel,
  distillModel,
  runTraining,
  runDistillation,
  runtime,
}: TensorflowIntegrationArgs,
deps: Partial<TensorflowIntegrationDeps> = {}) {
  const { useIntegration } = { useIntegration: useTrainingIntegration, ...deps };
  return useIntegration({
    useDatasetState,
    useTrainingRunsState,
    useUiState: useTensorflowUiStateWithDeps,
    useLogic: ({ dataset, trainingRuns, prependTrainingRun, ui }) =>
      useTensorflowLogic({
        dataset,
        trainingRuns,
        prependTrainingRun,
        ui,
        trainModel,
        distillModel,
        runTraining,
        runDistillation,
        runtime,
      }),
  });
}
