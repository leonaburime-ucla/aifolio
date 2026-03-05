import type {
  TensorflowDistillRequest,
  TensorflowTrainRequest,
} from "@/features/ml/__types__/typescript/api/tensorflowApi.types";
import type {
  RunTensorflowDistillationDeps,
  RunTensorflowDistillationProblem,
  RunTensorflowDistillationResult,
  RunTensorflowTrainingDeps,
  RunTensorflowTrainingProblem,
  RunTensorflowTrainingResult,
  TensorflowDistillModelResult,
  TensorflowTrainModelResult,
} from "@/features/ml/__types__/typescript/react/orchestrators/tensorflowTrainingOrchestrator.types";
import {
  runDistillation,
  runTrainingSweep,
} from "@/features/ml/typescript/react/orchestrators/training.orchestrator.shared";

export type {
  RunTensorflowDistillationDeps,
  RunTensorflowDistillationProblem,
  RunTensorflowDistillationResult,
  RunTensorflowTrainingDeps,
  RunTensorflowTrainingProblem,
  RunTensorflowTrainingResult,
  TensorflowTeacherConfig,
} from "@/features/ml/__types__/typescript/react/orchestrators/tensorflowTrainingOrchestrator.types";
export type { TensorflowTrainingMode } from "@/features/ml/__types__/typescript/api/tensorflowApi.types";

/**
 * Executes a full TensorFlow training sweep and records each run outcome via injected side effects.
 */
export function runTensorflowTraining(
  problem: RunTensorflowTrainingProblem,
  deps: RunTensorflowTrainingDeps
): Promise<RunTensorflowTrainingResult> {
  return runTrainingSweep<
    RunTensorflowTrainingProblem,
    TensorflowTrainRequest,
    TensorflowTrainModelResult
  >(problem, deps);
}

/**
 * Executes a single TensorFlow distillation run from a selected teacher configuration.
 */
export function runTensorflowDistillation(
  problem: RunTensorflowDistillationProblem,
  deps: RunTensorflowDistillationDeps
): Promise<RunTensorflowDistillationResult> {
  return runDistillation<
    RunTensorflowDistillationProblem,
    TensorflowDistillRequest,
    TensorflowDistillModelResult
  >(problem, deps, {
    // Keep distillation faster than full teacher training for interactive runs.
    resolveDistilledEpochs: (teacherEpochs) => Math.min(24, Math.max(8, Math.round(teacherEpochs * 0.4))),
  });
}
