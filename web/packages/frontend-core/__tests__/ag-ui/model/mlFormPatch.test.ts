import { describe, expect, it } from "vitest";
import {
  normalizeLookupToken,
  normalizeTrainingModeValue,
  normalizeDatasetIdValue,
  normalizeMlFormPatchAliases,
  resolveMlFormPatchFromToolArgs,
} from "../../../src/ag-ui/model/mlFormPatch";

describe("normalizeLookupToken", () => {
  it("lowercases and replaces non-alphanumeric with underscores", () => {
    expect(normalizeLookupToken("Neural Net")).toBe("neural_net");
  });

  it("replaces & with 'and'", () => {
    expect(normalizeLookupToken("wide & deep")).toBe("wide_and_deep");
  });

  it("trims leading/trailing underscores", () => {
    expect(normalizeLookupToken("  _hello_  ")).toBe("hello");
  });

  it("collapses multiple underscores", () => {
    expect(normalizeLookupToken("foo---bar___baz")).toBe("foo_bar_baz");
  });
});

describe("normalizeTrainingModeValue", () => {
  it("returns non-string values unchanged", () => {
    expect(normalizeTrainingModeValue(42)).toBe(42);
    expect(normalizeTrainingModeValue(null)).toBe(null);
  });

  it("resolves known aliases to canonical names", () => {
    expect(normalizeTrainingModeValue("neural_net")).toBe("mlp_dense");
    expect(normalizeTrainingModeValue("Neural Network")).toBe("mlp_dense");
    expect(normalizeTrainingModeValue("tab_resnet")).toBe("tabresnet");
    expect(normalizeTrainingModeValue("wide_deep")).toBe("wide_and_deep");
  });

  it("passes through supported modes unchanged", () => {
    expect(normalizeTrainingModeValue("tabresnet")).toBe("tabresnet");
    expect(normalizeTrainingModeValue("mlp_dense")).toBe("mlp_dense");
  });

  it("returns original value when not a known mode or alias", () => {
    expect(normalizeTrainingModeValue("unknown_model")).toBe("unknown_model");
  });
});

describe("normalizeDatasetIdValue", () => {
  it("returns non-string values unchanged", () => {
    expect(normalizeDatasetIdValue(123)).toBe(123);
  });

  it("returns empty-string values unchanged", () => {
    expect(normalizeDatasetIdValue("  ")).toBe("  ");
  });

  it("resolves known dataset aliases", () => {
    expect(normalizeDatasetIdValue("customer_churn")).toBe("customer_churn_telco.csv");
    expect(normalizeDatasetIdValue("fraud")).toBe("fraud_detection_phishing_websites.csv");
    expect(normalizeDatasetIdValue("house_prices")).toBe("house_prices_ames.csv");
  });

  it("trims and returns unknown datasets", () => {
    expect(normalizeDatasetIdValue("  custom_data.csv  ")).toBe("custom_data.csv");
  });
});

describe("normalizeMlFormPatchAliases", () => {
  it("resolves field aliases to canonical keys", () => {
    const patch = { epochs: "10", batchSize: "32" };
    const result = normalizeMlFormPatchAliases(patch);
    expect(result.epoch_values).toBe("10");
    expect(result.batch_sizes).toBe("32");
    expect(result.epochs).toBeUndefined();
    expect(result.batchSize).toBeUndefined();
  });

  it("does not overwrite existing canonical keys", () => {
    const patch = { epoch_values: "20", epochs: "10" };
    const result = normalizeMlFormPatchAliases(patch);
    expect(result.epoch_values).toBe("20");
  });

  it("normalizes training_mode via normalizeTrainingModeValue", () => {
    const patch = { training_mode: "neural_net" };
    const result = normalizeMlFormPatchAliases(patch);
    expect(result.training_mode).toBe("mlp_dense");
  });

  it("normalizes dataset_id via normalizeDatasetIdValue", () => {
    const patch = { dataset_id: "churn" };
    const result = normalizeMlFormPatchAliases(patch);
    expect(result.dataset_id).toBe("customer_churn_telco.csv");
  });
});

describe("resolveMlFormPatchFromToolArgs", () => {
  it("extracts fields from args.fields when it is an object", () => {
    const result = resolveMlFormPatchFromToolArgs({
      fields: { epochs: "5", batchSize: "16" },
    });
    expect(result.epoch_values).toBe("5");
    expect(result.batch_sizes).toBe("16");
  });

  it("falls back to top-level args when fields is not an object", () => {
    const result = resolveMlFormPatchFromToolArgs({
      epochs: "5",
      batchSize: "16",
    });
    expect(result.epoch_values).toBe("5");
    expect(result.batch_sizes).toBe("16");
  });

  it("promotes set_sweep_values to run_sweep when run_sweep is missing", () => {
    const result = resolveMlFormPatchFromToolArgs({
      set_sweep_values: true,
    });
    expect(result.run_sweep).toBe(true);
  });

  it("does not promote set_sweep_values when run_sweep is already set", () => {
    const result = resolveMlFormPatchFromToolArgs({
      set_sweep_values: true,
      runSweep: false,
    });
    expect(result.run_sweep).toBe(false);
  });
});
