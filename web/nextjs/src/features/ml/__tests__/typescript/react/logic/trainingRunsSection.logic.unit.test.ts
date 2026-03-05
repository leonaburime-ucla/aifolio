import { describe, expect, it } from "vitest";
import {
  buildDistillActionModel,
  resolveTeacherKey,
} from "@/features/ml/typescript/logic/trainingRunsSection.logic";

describe("trainingRunsSection.logic", () => {
  it("resolves teacher key preferring run_id, then model_id, then model_path", () => {
    expect(resolveTeacherKey({ row: { run_id: "run-1" } })).toBe("run-1");
    expect(resolveTeacherKey({ row: { run_id: "n/a", model_id: "model-1" } })).toBe("model-1");
    expect(resolveTeacherKey({ row: { model_path: "/tmp/model.pt" } })).toBe("/tmp/model.pt");
  });

  it("returns student_model action for distilled rows", () => {
    const action = buildDistillActionModel({
      row: { result: "distilled", run_id: "run-1" },
    });

    expect(action.kind).toBe("student_model");
  });

  it("returns not_available for unsupported or ineligible rows", () => {
    const unsupportedAction = buildDistillActionModel({
      row: { result: "completed", run_id: "run-1" },
      isDistillationSupportedForRun: () => false,
    });
    expect(unsupportedAction.kind).toBe("not_available");

    const ineligibleAction = buildDistillActionModel({
      row: { result: "failed", run_id: "run-1" },
    });
    expect(ineligibleAction.kind).toBe("not_available");
  });

  it("returns show_distilled when teacher already has distilled output", () => {
    const action = buildDistillActionModel({
      row: { result: "completed", run_id: "run-2" },
      distilledByTeacher: { "run-2": "distilled-run-id" },
    });

    expect(action.kind).toBe("show_distilled");
  });

  it("returns distill and marks active row when distillation is in progress", () => {
    const action = buildDistillActionModel({
      row: { result: "completed", run_id: "run-3" },
      distillingTeacherKey: "run-3",
    });

    expect(action.kind).toBe("distill");
    expect(action.isDistillingThisRow).toBe(true);
  });

  it("treats rows without result/teacher key as not available", () => {
    const action = buildDistillActionModel({
      row: { run_id: "n/a", model_id: "n/a", model_path: "n/a" },
    });

    expect(action.kind).toBe("not_available");
  });
});
