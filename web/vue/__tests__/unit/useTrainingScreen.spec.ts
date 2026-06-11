import { describe, it, expect, vi, beforeEach } from "vitest";
import { useTrainingScreen } from "~/features/ml/model";
import type { DatasetLoaderApi } from "~/features/ml/model";
import type { MlTrainingApi } from "~/features/ml/api";

function createMockDatasetApi(): DatasetLoaderApi {
  return {
    fetchManifest: vi.fn().mockResolvedValue({
      datasets: [{ id: "churn.csv", label: "Churn" }],
    }),
    fetchDataset: vi.fn().mockResolvedValue({
      rows: [{ col1: "a", col2: 1 }],
      columns: ["col1", "col2"],
    }),
  };
}

function createMockTrainingApi(): MlTrainingApi {
  return {
    trainPytorch: vi.fn().mockResolvedValue({ status: "ok", metrics: { test_metric: 0.9 } }),
    trainTensorflow: vi.fn().mockResolvedValue({ status: "ok", metrics: { test_metric: 0.88 } }),
    distillPytorch: vi.fn().mockResolvedValue({
      status: "ok",
      run_id: "student-run",
      metrics: { test_metric_name: "accuracy", test_metric_value: 0.86 },
    }),
    distillTensorflow: vi.fn().mockResolvedValue({
      status: "ok",
      run_id: "tf-student-run",
      metrics: { test_metric_name: "accuracy", test_metric_value: 0.84 },
    }),
  };
}

describe("useTrainingScreen", () => {
  let screen: ReturnType<typeof useTrainingScreen>;
  let datasetApi: DatasetLoaderApi;
  let trainingApi: MlTrainingApi;

  beforeEach(() => {
    datasetApi = createMockDatasetApi();
    trainingApi = createMockTrainingApi();
    screen = useTrainingScreen({
      baseUrl: "/api/ai",
      framework: "pytorch",
      defaultTrainingMode: "linear_glm_baseline",
      defaultExcludeColumns: "customerID",
      datasetApi,
      trainingApi,
    });
  });

  describe("initial state", () => {
    it("sets default training mode from options", () => {
      expect(screen.trainingMode.value).toBe("linear_glm_baseline");
    });

    it("sets default exclude columns from options", () => {
      expect(screen.excludeColumns.value).toBe("customerID");
    });

    it("defaults task to auto", () => {
      expect(screen.task.value).toBe("auto");
    });

    it("defaults form values", () => {
      expect(screen.epochValues.value).toBe("60");
      expect(screen.batchSizes.value).toBe("64");
      expect(screen.learningRates.value).toBe("0.001");
      expect(screen.testSizes.value).toBe("0.2");
      expect(screen.hiddenDims.value).toBe("128");
      expect(screen.numHiddenLayers.value).toBe("2");
      expect(screen.dropouts.value).toBe("0.1");
    });
  });

  describe("isLinearBaseline", () => {
    it("true when training mode is linear_glm_baseline", () => {
      screen.trainingMode.value = "linear_glm_baseline";
      expect(screen.isLinearBaseline.value).toBe(true);
    });

    it("false for other modes", () => {
      screen.trainingMode.value = "mlp_dense";
      expect(screen.isLinearBaseline.value).toBe(false);
    });
  });

  describe("plannedRunCount", () => {
    it("calculates cartesian product of comma-separated values", () => {
      screen.epochValues.value = "10,20";
      screen.batchSizes.value = "32,64";
      screen.learningRates.value = "0.001";
      screen.testSizes.value = "0.2";
      expect(screen.plannedRunCount.value).toBe(4);
    });

    it("returns 0 when a field is empty", () => {
      screen.epochValues.value = "";
      expect(screen.plannedRunCount.value).toBe(0);
    });
  });

  describe("isTrainDisabled", () => {
    it("disabled when no dataset selected", () => {
      screen.selectedDatasetId.value = null;
      expect(screen.isTrainDisabled.value).toBe(true);
    });

    it("disabled when planned runs is 0", () => {
      screen.selectedDatasetId.value = "churn.csv";
      screen.epochValues.value = "";
      expect(screen.isTrainDisabled.value).toBe(true);
    });

    it("enabled when dataset selected and runs > 0", () => {
      screen.selectedDatasetId.value = "churn.csv";
      screen.epochValues.value = "10";
      screen.batchSizes.value = "32";
      screen.learningRates.value = "0.001";
      screen.testSizes.value = "0.2";
      expect(screen.isTrainDisabled.value).toBe(false);
    });
  });

  describe("onTrain()", () => {
    it("parses form values and calls train", async () => {
      screen.selectedDatasetId.value = "churn.csv";
      screen.targetColumn.value = "Churn";
      screen.epochValues.value = "10,20";
      screen.batchSizes.value = "32";
      screen.learningRates.value = "0.001";
      screen.testSizes.value = "0.2";
      screen.hiddenDims.value = "128";
      screen.numHiddenLayers.value = "2";
      screen.dropouts.value = "0.1";
      screen.excludeColumns.value = "customerID,Order";
      screen.dateColumns.value = "Date";

      await screen.onTrain();

      expect(trainingApi.trainPytorch).toHaveBeenCalledTimes(2);
      expect(screen.trainingRuns.value).toHaveLength(2);
      expect(screen.trainingRuns.value[0].status).toBe("completed");
    });

    it("passes empty arrays when exclude/date columns are empty", async () => {
      screen.selectedDatasetId.value = "churn.csv";
      screen.targetColumn.value = "Churn";
      screen.excludeColumns.value = "";
      screen.dateColumns.value = "";

      await screen.onTrain();

      const call = (trainingApi.trainPytorch as any).mock.calls[0][0];
      expect(call.exclude_columns).toEqual([]);
      expect(call.date_columns).toEqual([]);
    });
  });

  describe("onCopyResults()", () => {
    it("writes TSV to clipboard", async () => {
      const writeText = vi.fn().mockResolvedValue(undefined);
      Object.assign(navigator, { clipboard: { writeText } });

      screen.selectedDatasetId.value = "churn.csv";
      screen.targetColumn.value = "Churn";
      await screen.onTrain();

      await screen.onCopyResults();
      expect(writeText).toHaveBeenCalled();
      const copied = writeText.mock.calls[0][0];
      expect(copied).toContain("completed_at\tdistill_action\tmetric_score");
      expect(copied).toContain("completed");
    });
  });

  describe("tensorflow framework", () => {
    it("uses trainTensorflow", async () => {
      const tfScreen = useTrainingScreen({
        baseUrl: "/api/ai",
        framework: "tensorflow",
        defaultTrainingMode: "wide_and_deep",
        datasetApi,
        trainingApi,
      });

      tfScreen.selectedDatasetId.value = "churn.csv";
      tfScreen.targetColumn.value = "Churn";
      await tfScreen.onTrain();

      expect(trainingApi.trainTensorflow).toHaveBeenCalled();
      expect(trainingApi.trainPytorch).not.toHaveBeenCalled();
    });

    it("supports wide_and_deep distillation without sending n/a model refs", async () => {
      const tfScreen = useTrainingScreen({
        baseUrl: "/api/ai",
        framework: "tensorflow",
        defaultTrainingMode: "wide_and_deep",
        datasetApi,
        trainingApi,
      });

      tfScreen.selectedDatasetId.value = "churn.csv";
      tfScreen.targetColumn.value = "Churn";

      expect(tfScreen.isDistillationSupportedForRun({ training_mode: "wide_and_deep" })).toBe(true);

      await tfScreen.onDistillFromRun({
        result: "completed",
        run_id: "teacher-run",
        model_id: "n/a",
        model_path: "n/a",
        target_column: "Churn",
        task: "classification",
        training_mode: "wide_and_deep",
        epochs: 12,
        batch_size: 32,
        learning_rate: 0.001,
        test_size: 0.2,
        hidden_dim: 128,
        num_hidden_layers: 2,
        dropout: 0.1,
      });

      expect(trainingApi.distillTensorflow).toHaveBeenCalledTimes(1);
      const payload = (trainingApi.distillTensorflow as any).mock.calls[0][0];
      expect(payload.teacher_run_id).toBe("teacher-run");
      expect(payload.teacher_model_id).toBeUndefined();
      expect(payload.teacher_model_path).toBeUndefined();
      expect(payload.training_mode).toBe("wide_and_deep");
      expect(payload.epochs).toBe(8);
    });
  });

  describe("dataset delegation", () => {
    it("loadManifest populates options", async () => {
      await screen.loadManifest();
      expect(screen.datasetOptions.value).toHaveLength(1);
      expect(screen.selectedDatasetId.value).toBe("churn.csv");
    });

    it("onDatasetChange updates selection", () => {
      screen.onDatasetChange("iris.csv");
      expect(screen.selectedDatasetId.value).toBe("iris.csv");
    });
  });
});
