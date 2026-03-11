import { describe, expect, it, vi } from "vitest";
import { createCopilotFrontendToolActions } from "@/features/ag-ui-chat/typescript/logic/copilotFrontendToolActions.logic";

describe("createCopilotFrontendToolActions", () => {
  it("wires action handlers and metadata", async () => {
    const handlers = {
      handleAddChartSpec: vi.fn((args) => ({ status: "ok", args })),
      handleClearCharts: vi.fn(() => "Cleared charts."),
      handleNavigateToPage: vi.fn((args) => `Navigated:${JSON.stringify(args)}`),
      handleSetActiveMlFormFields: vi.fn(async (args) => `Set active:${JSON.stringify(args)}`),
      handleChangeActiveMlTargetColumn: vi.fn(async (args) => `Change active:${JSON.stringify(args)}`),
      handleRandomizeActiveMlFormFields: vi.fn(async (args) => `Randomize active:${JSON.stringify(args)}`),
      handleStartActiveMlTrainingRuns: vi.fn(async () => "Started active."),
      handleStartPytorchTrainingRuns: vi.fn(async () => "Started pytorch."),
      handleTrainPytorchModel: vi.fn(async (args) => `Train pytorch:${JSON.stringify(args)}`),
      handleSetPytorchFormFields: vi.fn(async (args) => `Set pytorch:${JSON.stringify(args)}`),
      handleChangePytorchTargetColumn: vi.fn(async (args) => `Change pytorch:${JSON.stringify(args)}`),
      handleRandomizePytorchFormFields: vi.fn(async (args) => `Randomize pytorch:${JSON.stringify(args)}`),
      handleStartTensorflowTrainingRuns: vi.fn(async () => "Started tensorflow."),
      handleTrainTensorflowModel: vi.fn(async (args) => `Train tensorflow:${JSON.stringify(args)}`),
      handleSetTensorflowFormFields: vi.fn(async (args) => `Set tensorflow:${JSON.stringify(args)}`),
      handleChangeTensorflowTargetColumn: vi.fn(async (args) => `Change tensorflow:${JSON.stringify(args)}`),
      handleRandomizeTensorflowFormFields: vi.fn(async (args) => `Randomize tensorflow:${JSON.stringify(args)}`),
    };

    const actions = createCopilotFrontendToolActions(handlers);

    expect(actions.addChartSpec.name).toBe("add_chart_spec");
    expect(actions.setActiveMlFormFields.name).toBe("set_active_ml_form_fields");
    expect(actions.startActiveMlTrainingRuns.name).toBe("start_active_ml_training_runs");
    expect(actions.setPytorchFormFields.name).toBe("set_pytorch_form_fields");
    expect(actions.changePytorchTargetColumn.name).toBe("change_pytorch_target_column");
    expect(actions.setTensorflowFormFields.name).toBe("set_tensorflow_form_fields");
    expect(actions.changeTensorflowTargetColumn.name).toBe("change_tensorflow_target_column");

    await actions.setActiveMlFormFields.handler({ batch_sizes: [33, 40] });
    expect(handlers.handleSetActiveMlFormFields).toHaveBeenCalledWith({ batch_sizes: [33, 40] });

    await actions.trainPytorchModel.handler({ dataset_id: "d1", target_column: "y" });
    expect(handlers.handleTrainPytorchModel).toHaveBeenCalledWith({ dataset_id: "d1", target_column: "y" });

    await actions.randomizePytorchFormFields.handler({ style: "safe" });
    expect(handlers.handleRandomizePytorchFormFields).toHaveBeenCalledWith({ style: "safe" });

    await actions.changePytorchTargetColumn.handler({ mode: "different" });
    expect(handlers.handleChangePytorchTargetColumn).toHaveBeenCalledWith({ mode: "different" });

    await actions.trainTensorflowModel.handler({ dataset_id: "d2", target_column: "z" });
    expect(handlers.handleTrainTensorflowModel).toHaveBeenCalledWith({ dataset_id: "d2", target_column: "z" });

    await actions.randomizeTensorflowFormFields.handler({ style: "balanced" });
    expect(handlers.handleRandomizeTensorflowFormFields).toHaveBeenCalledWith({ style: "balanced" });

    await actions.changeTensorflowTargetColumn.handler({ target_column: "revenue" });
    expect(handlers.handleChangeTensorflowTargetColumn).toHaveBeenCalledWith({ target_column: "revenue" });
  });
});
