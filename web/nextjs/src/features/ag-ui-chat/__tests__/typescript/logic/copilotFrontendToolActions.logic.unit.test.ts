import { describe, expect, it, vi } from "vitest";
import { createCopilotFrontendToolActions } from "@/features/ag-ui-chat/typescript/logic/copilotFrontendToolActions.logic";

describe("createCopilotFrontendToolActions", () => {
  it("wires action handlers and metadata", async () => {
    const handlers = {
      handleAddChartSpec: vi.fn((args) => ({ status: "ok", args })),
      handleClearCharts: vi.fn(() => ({ status: "ok" as const, cleared: true as const })),
      handleNavigateToPage: vi.fn((args) => ({ status: "ok", args })),
      handleStartPytorchTrainingRuns: vi.fn(async () => ({ status: "ok" })),
      handleTrainPytorchModel: vi.fn(async (args) => ({ status: "ok", args })),
      handleSetPytorchFormFields: vi.fn(async (args) => ({ status: "ok", args })),
      handleRandomizePytorchFormFields: vi.fn(async (args) => ({ status: "ok", args })),
      handleStartTensorflowTrainingRuns: vi.fn(async () => ({ status: "ok" })),
      handleTrainTensorflowModel: vi.fn(async (args) => ({ status: "ok", args })),
      handleSetTensorflowFormFields: vi.fn(async (args) => ({ status: "ok", args })),
      handleRandomizeTensorflowFormFields: vi.fn(async (args) => ({ status: "ok", args })),
    };

    const actions = createCopilotFrontendToolActions(handlers);

    expect(actions.addChartSpec.name).toBe("add_chart_spec");
    expect(actions.setPytorchFormFields.name).toBe("set_pytorch_form_fields");
    expect(actions.setTensorflowFormFields.name).toBe("set_tensorflow_form_fields");

    await actions.trainPytorchModel.handler({ dataset_id: "d1", target_column: "y" });
    expect(handlers.handleTrainPytorchModel).toHaveBeenCalledWith({ dataset_id: "d1", target_column: "y" });

    await actions.randomizePytorchFormFields.handler({ style: "safe" });
    expect(handlers.handleRandomizePytorchFormFields).toHaveBeenCalledWith({ style: "safe" });

    await actions.trainTensorflowModel.handler({ dataset_id: "d2", target_column: "z" });
    expect(handlers.handleTrainTensorflowModel).toHaveBeenCalledWith({ dataset_id: "d2", target_column: "z" });

    await actions.randomizeTensorflowFormFields.handler({ style: "balanced" });
    expect(handlers.handleRandomizeTensorflowFormFields).toHaveBeenCalledWith({ style: "balanced" });
  });
});
