import { afterEach, describe, expect, it } from "vitest";
import {
  buildRandomPytorchFormPatch,
  buildRandomTensorflowFormPatch,
  handleChangePytorchTargetColumn,
  handleChangeTensorflowTargetColumn,
} from "@/features/ml/typescript/ai/agUi/mlTrainingTools.adapter";

function expectArrayWithCount(value: unknown, count: number): void {
  expect(Array.isArray(value)).toBe(true);
  expect((value as unknown[]).length).toBe(count);
}

function renderSelect(
  dataAiField: string,
  values: string[],
  selectedValue: string
): void {
  const options = values
    .map((value) => `<option value="${value}"${value === selectedValue ? " selected" : ""}>${value}</option>`)
    .join("");
  document.body.insertAdjacentHTML(
    "beforeend",
    `<select data-ai-field="${dataAiField}">${options}</select>`
  );
}

afterEach(() => {
  document.body.innerHTML = "";
  delete (window as Window & {
    __AIFOLIO_TENSORFLOW_FORM_BRIDGE__?: unknown;
  }).__AIFOLIO_TENSORFLOW_FORM_BRIDGE__;
});

describe("mlTrainingTools randomize builders", () => {
  it("defaults PyTorch randomize sweeps to one value per field when value_count is omitted", () => {
    const patch = buildRandomPytorchFormPatch({
      confirm_randomize: true,
      style: "balanced",
    });

    expectArrayWithCount(patch.epoch_values, 1);
    expectArrayWithCount(patch.batch_sizes, 1);
    expectArrayWithCount(patch.learning_rates, 1);
    expectArrayWithCount(patch.test_sizes, 1);
    expectArrayWithCount(patch.hidden_dims, 1);
    expectArrayWithCount(patch.num_hidden_layers, 1);
    expectArrayWithCount(patch.dropouts, 1);
  });

  it("respects value_count=1 for PyTorch sweep fields", () => {
    const patch = buildRandomPytorchFormPatch({
      confirm_randomize: true,
      value_count: 1,
      style: "balanced",
    });

    expectArrayWithCount(patch.epoch_values, 1);
    expectArrayWithCount(patch.batch_sizes, 1);
    expectArrayWithCount(patch.learning_rates, 1);
    expectArrayWithCount(patch.test_sizes, 1);
    expectArrayWithCount(patch.hidden_dims, 1);
    expectArrayWithCount(patch.num_hidden_layers, 1);
    expectArrayWithCount(patch.dropouts, 1);
  });

  it("defaults TensorFlow randomize sweeps to one value per field when value_count is omitted", () => {
    const patch = buildRandomTensorflowFormPatch({
      confirm_randomize: true,
      style: "balanced",
    });

    expectArrayWithCount(patch.epoch_values, 1);
    expectArrayWithCount(patch.batch_sizes, 1);
    expectArrayWithCount(patch.learning_rates, 1);
    expectArrayWithCount(patch.test_sizes, 1);
    expectArrayWithCount(patch.hidden_dims, 1);
    expectArrayWithCount(patch.num_hidden_layers, 1);
    expectArrayWithCount(patch.dropouts, 1);
  });

  it("respects value_count=1 for TensorFlow sweep fields", () => {
    const patch = buildRandomTensorflowFormPatch({
      confirm_randomize: true,
      value_count: 1,
      style: "balanced",
    });

    expectArrayWithCount(patch.epoch_values, 1);
    expectArrayWithCount(patch.batch_sizes, 1);
    expectArrayWithCount(patch.learning_rates, 1);
    expectArrayWithCount(patch.test_sizes, 1);
    expectArrayWithCount(patch.hidden_dims, 1);
    expectArrayWithCount(patch.num_hidden_layers, 1);
    expectArrayWithCount(patch.dropouts, 1);
  });

  it("still allows explicit multi-value TensorFlow sweeps", () => {
    const patch = buildRandomTensorflowFormPatch({
      confirm_randomize: true,
      value_count: 2,
      style: "balanced",
    });

    expectArrayWithCount(patch.epoch_values, 2);
    expectArrayWithCount(patch.batch_sizes, 2);
    expectArrayWithCount(patch.learning_rates, 2);
    expectArrayWithCount(patch.test_sizes, 2);
    expectArrayWithCount(patch.hidden_dims, 2);
    expectArrayWithCount(patch.num_hidden_layers, 2);
    expectArrayWithCount(patch.dropouts, 2);
  });

  it("does not mutate the PyTorch target column during randomization", () => {
    document.body.innerHTML = "";
    renderSelect("pytorch_target_column", ["churn", "revenue", "segment"], "churn");
    renderSelect("pytorch_task", ["auto", "classification"], "classification");

    const patch = buildRandomPytorchFormPatch({
      confirm_randomize: true,
      randomize_model_fields: true,
    });

    expect(patch.target_column).toBeUndefined();
  });

  it("does not mutate the TensorFlow target column during randomization and preserves task when not randomizing model fields", () => {
    document.body.innerHTML = "";
    renderSelect("tensorflow_target_column", ["churn", "revenue", "segment"], "churn");
    renderSelect("tensorflow_task", ["auto", "classification"], "classification");

    const randomizeModelFieldsPatch = buildRandomTensorflowFormPatch({
      confirm_randomize: true,
      randomize_model_fields: true,
    });
    const preserveTaskPatch = buildRandomTensorflowFormPatch({
      confirm_randomize: true,
      randomize_model_fields: false,
    });

    expect(randomizeModelFieldsPatch.target_column).toBeUndefined();
    expect(preserveTaskPatch.task).toBe("classification");
  });
});

describe("mlTrainingTools target-column handlers", () => {
  it("changes the PyTorch target column to a different option by default", () => {
    document.body.innerHTML = "";
    renderSelect("pytorch_target_column", ["churn", "revenue", "segment"], "churn");

    const result = handleChangePytorchTargetColumn();
    const select = document.querySelector('[data-ai-field="pytorch_target_column"]') as HTMLSelectElement;

    expect(result).toEqual({
      status: "ok",
      applied: ["target_column"],
      skipped: [],
    });
    expect(select.value).not.toBe("churn");
  });

  it("changes the TensorFlow target column through the registered bridge", () => {
    document.body.innerHTML = "";
    renderSelect("tensorflow_target_column", ["churn", "revenue", "segment"], "churn");
    const bridge = {
      applyPatch: (patch: { target_column?: string }) => {
        const select = document.querySelector('[data-ai-field="tensorflow_target_column"]') as HTMLSelectElement;
        if (patch.target_column) {
          select.value = patch.target_column;
        }
        return { applied: patch.target_column ? ["target_column"] : [], skipped: [] };
      },
    };
    (window as Window & {
      __AIFOLIO_TENSORFLOW_FORM_BRIDGE__?: typeof bridge;
    }).__AIFOLIO_TENSORFLOW_FORM_BRIDGE__ = bridge;

    const result = handleChangeTensorflowTargetColumn({ mode: "next" });
    const select = document.querySelector('[data-ai-field="tensorflow_target_column"]') as HTMLSelectElement;

    expect(result).toEqual({
      status: "ok",
      applied: ["target_column"],
      skipped: [],
      via: "bridge",
    });
    expect(select.value).toBe("revenue");
    delete (window as Window & {
      __AIFOLIO_TENSORFLOW_FORM_BRIDGE__?: typeof bridge;
    }).__AIFOLIO_TENSORFLOW_FORM_BRIDGE__;
  });
});
