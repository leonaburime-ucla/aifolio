import type { PytorchDistillRequest, PytorchTrainRequest } from "@/features/ml/__types__/typescript/api/pytorchApi.types";
import type {
  PytorchDistillModelResult,
  PytorchTrainModelResult,
  RunPytorchDistillationDeps,
  RunPytorchDistillationProblem,
  RunPytorchDistillationResult,
  RunPytorchTrainingDeps,
  RunPytorchTrainingProblem,
  RunPytorchTrainingResult,
} from "@/features/ml/__types__/typescript/react/orchestrators/pytorchTrainingOrchestrator.types";
import {
  runDistillation,
  runTrainingSweep,
} from "@/features/ml/typescript/react/orchestrators/training.orchestrator.shared";

export type {
  PytorchTeacherConfig,
  RunPytorchDistillationDeps,
  RunPytorchDistillationProblem,
  RunPytorchDistillationResult,
  RunPytorchTrainingDeps,
  RunPytorchTrainingProblem,
  RunPytorchTrainingResult,
} from "@/features/ml/__types__/typescript/react/orchestrators/pytorchTrainingOrchestrator.types";
export type { PytorchTrainingMode } from "@/features/ml/__types__/typescript/api/pytorchApi.types";

/**
 * Executes a full PyTorch training sweep and records each run outcome via injected side effects.
 */
export function runPytorchTraining(
  problem: RunPytorchTrainingProblem,
  deps: RunPytorchTrainingDeps
): Promise<RunPytorchTrainingResult> {
  return runTrainingSweep<
    RunPytorchTrainingProblem,
    PytorchTrainRequest,
    PytorchTrainModelResult
  >(problem, deps);
}

/**
 * Executes a single PyTorch distillation run from a selected teacher configuration.
 */
export function runPytorchDistillation(
  problem: RunPytorchDistillationProblem,
  deps: RunPytorchDistillationDeps
): Promise<RunPytorchDistillationResult> {
  return runDistillation<
    RunPytorchDistillationProblem,
    PytorchDistillRequest,
    PytorchDistillModelResult
  >(problem, deps, {
    resolveDistilledEpochs: (teacherEpochs) => Math.max(30, Math.round(teacherEpochs)),
  });
}
