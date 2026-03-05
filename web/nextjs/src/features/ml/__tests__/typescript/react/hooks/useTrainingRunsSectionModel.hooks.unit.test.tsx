import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { renderHook } from "@testing-library/react";
import { useTrainingRunsSectionModel } from "@/features/ml/typescript/react/hooks/useTrainingRunsSectionModel.hooks";

describe("useTrainingRunsSectionModel", () => {
  it("returns table height and distill button renderer", () => {
    const onDistillFromRun = vi.fn();
    const { result } = renderHook(() =>
      useTrainingRunsSectionModel({
        trainingRuns: [{ run_id: "r1", result: "completed", training_mode: "mlp_dense" }],
        onDistillFromRun,
      })
    );

    expect(result.current.trainingTableHeight).toBeGreaterThan(0);

    const row = { run_id: "r1", model_id: "m1", result: "completed", training_mode: "mlp_dense" };
    render(<div>{result.current.cellRenderers.distill_action(null, row)}</div>);
    fireEvent.click(screen.getByRole("button", { name: "Distill" }));
    expect(onDistillFromRun).toHaveBeenCalledWith(row);
  });

  it("renders student/show/not-available variants", () => {
    const onSeeDistilledFromRun = vi.fn();
    const row = { run_id: "teacher-1", model_id: "m1", result: "completed", training_mode: "mlp_dense" };

    const { result: studentModel } = renderHook(() =>
      useTrainingRunsSectionModel({
        trainingRuns: [row],
        distilledByTeacher: { "teacher-1": "distilled-1" },
      })
    );
    render(<div>{studentModel.current.cellRenderers.distill_action(null, { ...row, result: "distilled" })}</div>);
    expect(screen.getByText("Student Model")).toBeInTheDocument();

    const { result: showModel } = renderHook(() =>
      useTrainingRunsSectionModel({
        trainingRuns: [row],
        onSeeDistilledFromRun,
        distilledByTeacher: { "teacher-1": "distilled-1" },
      })
    );
    render(<div>{showModel.current.cellRenderers.distill_action(null, row)}</div>);
    fireEvent.click(screen.getByRole("button", { name: "Show Distilled" }));
    expect(onSeeDistilledFromRun).toHaveBeenCalledWith(row);

    const { result: notAvailable } = renderHook(() =>
      useTrainingRunsSectionModel({
        trainingRuns: [row],
        isDistillationSupportedForRun: () => false,
      })
    );
    render(<div>{notAvailable.current.cellRenderers.distill_action(null, row)}</div>);
    expect(screen.getByText("Not Available")).toBeInTheDocument();
  });

  it("renders a disabled distilling state for the active teacher row", () => {
    const row = { run_id: "teacher-9", model_id: "m9", result: "completed", training_mode: "mlp_dense" };
    const onDistillFromRun = vi.fn();

    const { result } = renderHook(() =>
      useTrainingRunsSectionModel({
        trainingRuns: [row],
        onDistillFromRun,
        distillingTeacherKey: "teacher-9",
      })
    );

    render(<div>{result.current.cellRenderers.distill_action(null, row)}</div>);
    expect(screen.getByRole("button", { name: "Distilling..." })).toBeDisabled();
  });
});
