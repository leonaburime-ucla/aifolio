import { describe, expect, it } from "vitest";
import {
  handleNavigateToPage,
  resolvePytorchFormPatchFromToolArgs,
  resolveTensorflowFormPatchFromToolArgs,
} from "@/features/ag-ui-chat/typescript/logic/frontendTools.logic";

describe("resolvePytorchFormPatchFromToolArgs", () => {
  it("prefers nested fields object when provided", () => {
    const result = resolvePytorchFormPatchFromToolArgs({
      fields: { epoch_values: [10, 20], run_sweep: true },
      ignored: "value",
    });

    expect(result).toEqual({ epoch_values: [10, 20], run_sweep: true });
  });

  it("falls back to root args object when fields is not an object", () => {
    const result = resolvePytorchFormPatchFromToolArgs({
      epoch_values: [30, 60],
      run_sweep: false,
    });

    expect(result).toEqual({ epoch_values: [30, 60], run_sweep: false });
  });

  it("normalizes singular and camelCase aliases into canonical PyTorch sweep keys", () => {
    const result = resolvePytorchFormPatchFromToolArgs({
      fields: {
        batch_size: [33, 40],
        hiddenDim: [50, 75],
        testSize: [0.25, 0.3],
      },
    });

    expect(result).toEqual({
      batch_sizes: [33, 40],
      hidden_dims: [50, 75],
      test_sizes: [0.25, 0.3],
    });
  });

  it("normalizes common training-mode aliases into the canonical training_mode field", () => {
    const result = resolvePytorchFormPatchFromToolArgs({
      fields: {
        algorithm: "TabResNet",
      },
    });

    expect(result).toEqual({
      training_mode: "tabresnet",
    });
  });

  it("resolves friendly dataset names into canonical dataset ids", () => {
    const result = resolvePytorchFormPatchFromToolArgs({
      fields: {
        dataset: "fraud detection",
      },
    });

    expect(result).toEqual({
      dataset_id: "fraud_detection_phishing_websites.csv",
    });
  });
});

describe("handleNavigateToPage", () => {
  it("returns ok for supported alias", () => {
    const result = handleNavigateToPage("pytorch");
    expect(result).toEqual({ status: "ok", resolvedRoute: "/ml/pytorch" });
  });

  it("returns error for unsupported route", () => {
    const result = handleNavigateToPage("/not-real");
    expect(result.status).toBe("error");
  });
});

describe("resolveTensorflowFormPatchFromToolArgs", () => {
  it("prefers nested fields object when provided", () => {
    const result = resolveTensorflowFormPatchFromToolArgs({
      fields: { epoch_values: [12, 24], run_sweep: true },
      ignored: "value",
    });

    expect(result).toEqual({ epoch_values: [12, 24], run_sweep: true });
  });

  it("falls back to root args object when fields is not an object", () => {
    const result = resolveTensorflowFormPatchFromToolArgs({
      epoch_values: [16, 32],
      run_sweep: false,
    });

    expect(result).toEqual({ epoch_values: [16, 32], run_sweep: false });
  });

  it("normalizes TensorFlow aliases from root args into canonical sweep keys", () => {
    const result = resolveTensorflowFormPatchFromToolArgs({
      batchSize: [33, 40],
      hidden_dim: [50, 75],
      test_size: [0.25, 0.3],
    });

    expect(result).toEqual({
      batch_sizes: [33, 40],
      hidden_dims: [50, 75],
      test_sizes: [0.25, 0.3],
    });
  });

  it("maps architecture-style prompts onto TensorFlow training_mode values", () => {
    const result = resolveTensorflowFormPatchFromToolArgs({
      fields: {
        architecture: "wide and deep",
      },
    });

    expect(result).toEqual({
      training_mode: "wide_and_deep",
    });
  });

  it("resolves dataset labels for TensorFlow active-tab prompts", () => {
    const result = resolveTensorflowFormPatchFromToolArgs({
      fields: {
        dataset: "customer churn",
      },
    });

    expect(result).toEqual({
      dataset_id: "customer_churn_telco.csv",
    });
  });
});
