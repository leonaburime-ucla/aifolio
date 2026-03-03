import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { ReactNode } from "react";

vi.mock("@/core/views/components/General/Modal", () => ({
  Modal: ({ isOpen, title, children }: { isOpen: boolean; title: string; children: ReactNode }) =>
    isOpen ? (
      <div data-testid="modal">
        <h2>{title}</h2>
        {children}
      </div>
    ) : null,
}));

import {
  DistillMetricsModal,
  OptimalParamsModal,
} from "@/features/ml/typescript/react/views/components/MlTrainingModals";

describe("MlTrainingModals", () => {
  it("renders optimal params modal with prediction and apply button", () => {
    const onApply = vi.fn();
    const onClose = vi.fn();
    render(
      <OptimalParamsModal
        isOpen
        onClose={onClose}
        onApply={onApply}
        activeAlgorithm="mlp_dense"
        pendingOptimalParams={{
          epochs: 10,
          learning_rate: 0.00123,
          test_size: 0.2,
          batch_size: 32,
          hidden_dim: 64,
          num_hidden_layers: 2,
          dropout: 0.1,
        }}
        pendingOptimalPrediction={{ metricName: "accuracy", metricValue: 0.95 }}
      />
    );

    expect(screen.getByText("Bayesian Optimization Suggestion")).toBeInTheDocument();
    expect(screen.getByText(/Predicted: accuracy/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Update Table With Values" }));
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(onApply).toHaveBeenCalledTimes(1);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("renders distill metrics and fallback copy when no model files", () => {
    const onClose = vi.fn();
    render(
      <DistillMetricsModal
        isOpen
        onClose={onClose}
        distillMetrics={{ test_metric_name: "accuracy", test_metric_value: 0.8 }}
        distillModelId={null}
        distillModelPath={null}
        distillComparison={{
          metricName: "accuracy",
          teacherMetricValue: 0.82,
          studentMetricValue: 0.8,
          qualityDelta: -0.02,
          higherIsBetter: true,
          teacherTrainingMode: "mlp_dense",
          studentTrainingMode: "mlp_dense",
          teacherHiddenDim: 128,
          studentHiddenDim: 32,
          teacherNumHiddenLayers: 3,
          studentNumHiddenLayers: 2,
          teacherInputDim: 20,
          studentInputDim: 20,
          teacherOutputDim: 2,
          studentOutputDim: 2,
          teacherModelSizeBytes: 1024,
          studentModelSizeBytes: 512,
          sizeSavedBytes: 512,
          sizeSavedPercent: null,
          teacherParamCount: 5000,
          studentParamCount: 2500,
          paramSavedCount: 2500,
          paramSavedPercent: null,
        }}
      />
    );

    expect(screen.getByText("Teacher vs Student")).toBeInTheDocument();
    expect(screen.getByText(/Model files were not saved for this run/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Close" }));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("renders model id/path block and disabled apply without params", () => {
    render(
      <>
        <OptimalParamsModal
          isOpen
          onClose={vi.fn()}
          onApply={vi.fn()}
          pendingOptimalParams={null}
          pendingOptimalPrediction={null}
        />
        <DistillMetricsModal
          isOpen
          onClose={vi.fn()}
          distillMetrics={null}
          distillModelId="model-x"
          distillModelPath="/tmp/model-x"
          distillComparison={null}
        />
      </>
    );

    const applyButtons = screen.getAllByRole("button", { name: "Update Table With Values" });
    expect(applyButtons.at(-1)).toBeDisabled();
    expect(screen.getByText("model-x")).toBeInTheDocument();
    expect(screen.getByText("/tmp/model-x")).toBeInTheDocument();
  });

  it("renders numeric size/param percentages when available", () => {
    render(
      <DistillMetricsModal
        isOpen
        onClose={vi.fn()}
        distillMetrics={{ test_metric_name: "f1", test_metric_value: 0.75 }}
        distillModelId={null}
        distillModelPath={null}
        distillComparison={{
          metricName: "f1",
          teacherMetricValue: 0.8,
          studentMetricValue: 0.75,
          qualityDelta: -0.05,
          higherIsBetter: true,
          teacherTrainingMode: "mlp_dense",
          studentTrainingMode: "mlp_dense",
          teacherHiddenDim: 128,
          studentHiddenDim: 64,
          teacherNumHiddenLayers: 3,
          studentNumHiddenLayers: 2,
          teacherInputDim: 20,
          studentInputDim: 20,
          teacherOutputDim: 2,
          studentOutputDim: 2,
          teacherModelSizeBytes: 4096,
          studentModelSizeBytes: 2048,
          sizeSavedBytes: 2048,
          sizeSavedPercent: 50,
          teacherParamCount: 9000,
          studentParamCount: 4500,
          paramSavedCount: 4500,
          paramSavedPercent: 50,
        }}
      />
    );

    expect(screen.getAllByText("(50%)").length).toBeGreaterThanOrEqual(2);
  });

  it("renders lower-is-better copy and model artifact n/a fallbacks", () => {
    render(
      <DistillMetricsModal
        isOpen
        onClose={vi.fn()}
        distillMetrics={{}}
        distillModelId={null}
        distillModelPath="/tmp/model-only"
        distillComparison={{
          metricName: "loss",
          teacherMetricValue: 0.3,
          studentMetricValue: 0.2,
          qualityDelta: 0.1,
          higherIsBetter: false,
          teacherTrainingMode: null,
          studentTrainingMode: null,
          teacherHiddenDim: null,
          studentHiddenDim: null,
          teacherNumHiddenLayers: null,
          studentNumHiddenLayers: null,
          teacherInputDim: null,
          studentInputDim: null,
          teacherOutputDim: null,
          studentOutputDim: null,
          teacherModelSizeBytes: null,
          studentModelSizeBytes: null,
          sizeSavedBytes: null,
          sizeSavedPercent: null,
          teacherParamCount: null,
          studentParamCount: null,
          paramSavedCount: null,
          paramSavedPercent: null,
        }}
      />
    );

    expect(screen.getByText("(lower is better)")).toBeInTheDocument();
    expect(screen.getAllByText(/model_id:/).length).toBeGreaterThan(0);
    expect(screen.getByText("/tmp/model-only")).toBeInTheDocument();
  });
});
