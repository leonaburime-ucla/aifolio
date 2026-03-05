import type { TrainingRunRow } from "@/features/ml/__types__/typescript/utils/trainingRuns.types";

export type DistillActionKind =
  | "student_model"
  | "not_available"
  | "show_distilled"
  | "distill";

export type DistillActionModel = {
  kind: DistillActionKind;
  teacherKey: string;
  isDistillingThisRow: boolean;
};

export type BuildDistillActionModelParams = {
  row: TrainingRunRow;
  isDistillationSupportedForRun?: (row: TrainingRunRow) => boolean;
  distillingTeacherKey?: string | null;
  distilledByTeacher?: Record<string, string>;
};

/**
 * Resolves the stable teacher identifier used across distillation actions.
 * @param params - Required parameters.
 * @param params.row - Training run row.
 * @returns Teacher key string or empty string when unavailable.
 */
export function resolveTeacherKey({
  row,
}: {
  row: TrainingRunRow;
}): string {
  const runId = String(row.run_id ?? "");
  const modelId = String(row.model_id ?? "");
  const modelPath = String(row.model_path ?? "");

  return (
    (runId && runId !== "n/a" ? runId : "") ||
    (modelId && modelId !== "n/a" ? modelId : "") ||
    (modelPath && modelPath !== "n/a" ? modelPath : "")
  );
}

/**
 * Builds a pure, testable action model for the distillation cell.
 * @param params - Required parameters.
 * @returns Distillation action view model.
 */
export function buildDistillActionModel({
  row,
  isDistillationSupportedForRun,
  distillingTeacherKey = null,
  distilledByTeacher = {},
}: BuildDistillActionModelParams): DistillActionModel {
  const rowResult = String(row.result ?? "");
  const teacherKey = resolveTeacherKey({ row });
  const isEligibleTeacher = Boolean(teacherKey) && rowResult === "completed";
  const isSupportedMode = isDistillationSupportedForRun
    ? isDistillationSupportedForRun(row)
    : true;
  const isDistillingThisRow = distillingTeacherKey === teacherKey;
  const hasDistilled = Boolean(distilledByTeacher[teacherKey]);

  if (rowResult === "distilled") {
    return { kind: "student_model", teacherKey, isDistillingThisRow };
  }
  if (!isEligibleTeacher || !isSupportedMode) {
    return { kind: "not_available", teacherKey, isDistillingThisRow };
  }
  if (hasDistilled) {
    return { kind: "show_distilled", teacherKey, isDistillingThisRow };
  }
  return { kind: "distill", teacherKey, isDistillingThisRow };
}
