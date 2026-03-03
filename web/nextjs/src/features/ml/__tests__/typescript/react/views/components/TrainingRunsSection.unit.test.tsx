import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

const mockedUseModel = vi.fn();

vi.mock("@/features/ml/typescript/react/hooks/useTrainingRunsSectionModel.hooks", () => ({
  useTrainingRunsSectionModel: (params: unknown) => mockedUseModel(params),
}));

vi.mock("@/core/views/components/Datatable/DataTable", () => ({
  default: ({ rows }: { rows: Array<unknown> }) => <div data-testid="data-table">rows:{rows.length}</div>,
}));

import { TrainingRunsSection } from "@/features/ml/typescript/react/views/components/TrainingRunsSection";

describe("TrainingRunsSection", () => {
  it("renders empty state and disabled actions", () => {
    mockedUseModel.mockReturnValue({
      trainingTableHeight: 300,
      cellRenderers: { distill_action: () => <span>x</span> },
    });

    render(
      <TrainingRunsSection
        trainingRuns={[]}
        onCopyTrainingRuns={vi.fn()}
        onClearTrainingRuns={vi.fn()}
        onStopTrainingRuns={vi.fn()}
      />
    );

    expect(screen.getByText("No runs yet. Train once to populate the results table.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Copy Results" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Clear Runs" })).toBeDisabled();
  });

  it("renders table and stop requested message", () => {
    const onStop = vi.fn();
    mockedUseModel.mockReturnValue({
      trainingTableHeight: 300,
      cellRenderers: { distill_action: () => <span>x</span> },
    });

    render(
      <TrainingRunsSection
        trainingRuns={[{ run_id: "r1" }]}
        isTraining
        isStopRequested
        copyRunsStatus="Copied"
        onCopyTrainingRuns={vi.fn()}
        onClearTrainingRuns={vi.fn()}
        onStopTrainingRuns={onStop}
      />
    );

    expect(screen.getByTestId("data-table")).toHaveTextContent("rows:1");
    expect(screen.getByText("Copied")).toBeInTheDocument();
    expect(screen.getByText(/Stop requested\./)).toBeInTheDocument();

    const stopButton = screen.getByRole("button", { name: "Stop Requested..." });
    expect(stopButton).toBeDisabled();
    fireEvent.click(stopButton);
    expect(onStop).not.toHaveBeenCalled();
  });
});
