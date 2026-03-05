import type { PytorchTrainingMode } from "@/features/ml/__types__/typescript/api/pytorchApi.types";
import { useTrainingIntegration } from "@/features/ml/typescript/react/hooks/training.hooks";
import {
  useTrainingFrameworkLogic,
  useTrainingFrameworkUiState,
} from "@/features/ml/typescript/react/hooks/trainingFramework.hooks";
import type {
  PytorchIntegrationDeps,
  PytorchIntegrationArgs,
  PytorchLogicDeps,
  PytorchLogicArgs,
  PytorchUiStateDeps,
} from "@/features/ml/__types__/typescript/react/hooks/pytorchTraining.types";
export type {
  PytorchIntegrationArgs,
  PytorchLogicArgs,
  PytorchRuntimeDeps,
} from "@/features/ml/__types__/typescript/react/hooks/pytorchTraining.types";

export type { PytorchTrainingMode };

/**
 * PyTorch-specific training hooks module.
 *
 * Relative to other hooks in this folder:
 * - `trainingFramework.hooks.ts` owns shared framework-agnostic training behavior.
 * - This file only supplies PyTorch adapters and exported wrapper hook names.
 */

const PYTORCH_DISTILL_SUPPORTED_MODES: PytorchTrainingMode[] = [
  "mlp_dense",
  "linear_glm_baseline",
  "tabresnet",
];

/**
 * Type-guard for distillation-supported PyTorch modes.
 *
 * @param mode Raw training mode token.
 * @returns True when the provided mode supports current distillation flow.
 */
function isPytorchDistillSupportedMode(mode: string): mode is PytorchTrainingMode {
  return PYTORCH_DISTILL_SUPPORTED_MODES.includes(mode as PytorchTrainingMode);
}

/**
 * Creates PyTorch UI state by extending shared ML training UI state with
 * framework-specific `trainingMode`.
 *
 * @returns PyTorch UI state slice with training mode controls.
 */
export function usePytorchUiState() {
  return usePytorchUiStateWithDeps();
}

/**
 * Creates PyTorch UI state with injectable shared-hook dependency.
 *
 * @param deps Dependency overrides for UI state composition.
 * @param deps.useFrameworkUiState Shared framework UI-state hook.
 * @returns PyTorch UI state slice with training mode controls.
 */
export function usePytorchUiStateWithDeps(
  deps: Partial<PytorchUiStateDeps> = {}
) {
  const { useFrameworkUiState } = { useFrameworkUiState: useTrainingFrameworkUiState, ...deps };
  return useFrameworkUiState<PytorchTrainingMode>("mlp_dense");
}

/**
 * Core PyTorch training workflow logic.
 * Handles dataset defaults, sweep planning, training execution, and distillation flows.
 *
 * @param args Logic dependencies (dataset state, run state, ui state, orchestrators, runtime deps).
 * @param deps Optional dependency overrides for shared framework logic composition.
 * @returns PyTorch training behavior API consumed by integration/page layers.
 */
export function usePytorchLogic(
  args: PytorchLogicArgs,
  deps: Partial<PytorchLogicDeps> = {}
) {
  const { useFrameworkLogic } = { useFrameworkLogic: useTrainingFrameworkLogic, ...deps };
  return useFrameworkLogic({
    ...args,
    framework: {
      isDistillationSupportedMode: isPytorchDistillSupportedMode,
      trainModel: args.trainModel,
      distillModel: args.distillModel,
    },
  });
}

/**
 * Public integration hook consumed by pages/components.
 * Wires dataset state, run-store state, UI state, and PyTorch logic into one surface.
 *
 * @param args Injected dataset/runs/model/training orchestration dependencies.
 * @param deps Optional dependency overrides for integration composition.
 * @returns Unified PyTorch training integration model for UI composition.
 */
export function usePytorchTrainingIntegration({
  useDatasetState,
  useTrainingRunsState,
  trainModel,
  distillModel,
  runTraining,
  runDistillation,
  runtime,
}: PytorchIntegrationArgs,
deps: Partial<PytorchIntegrationDeps> = {}) {
  const { useIntegration } = { useIntegration: useTrainingIntegration, ...deps };
  return useIntegration({
    useDatasetState,
    useTrainingRunsState,
    useUiState: usePytorchUiStateWithDeps,
    useLogic: ({ dataset, trainingRuns, prependTrainingRun, ui }) =>
      usePytorchLogic({
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
