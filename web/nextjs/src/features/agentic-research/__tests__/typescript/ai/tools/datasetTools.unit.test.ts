import { describe, expect, it, vi } from "vitest";
import { handleAgenticSetActiveDataset } from "@/features/agentic-research/typescript/ai/tools/datasetTools";

describe("agentic datasetTools", () => {
  it("accepts exact id and preserves existing charts while setting dataset", () => {
    const setDatasetFn = vi.fn();

    expect(
      handleAgenticSetActiveDataset(
        "iris.csv",
        [{ id: "iris.csv", label: "Iris CSV" }],
        setDatasetFn
      )
    ).toEqual({ status: "ok", active_dataset_id: "iris.csv" });

    expect(setDatasetFn).toHaveBeenCalledWith("iris.csv");
  });

  it("resolves dataset by normalized/fuzzy label tokens", () => {
    const setDatasetFn = vi.fn();
    const manifest = [
      { id: "iris.csv", label: "Iris Dataset" },
      { id: "wine-quality.csv", label: "Wine Quality" },
    ];

    expect(
      handleAgenticSetActiveDataset(
        "wine quality",
        manifest,
        setDatasetFn
      )
    ).toEqual({ status: "ok", active_dataset_id: "wine-quality.csv" });

    expect(
      handleAgenticSetActiveDataset(
        "iri",
        manifest,
        setDatasetFn
      )
    ).toEqual({ status: "ok", active_dataset_id: "iris.csv" });
  });

  it("returns INVALID_DATASET_ID for unknown/blank ids", () => {
    const manifest = [{ id: "iris.csv", label: "Iris Dataset" }];

    expect(
      handleAgenticSetActiveDataset("", manifest, vi.fn())
    ).toEqual({
      status: "error",
      code: "INVALID_DATASET_ID",
      dataset_id: "",
      allowed_dataset_ids: ["iris.csv"],
    });

    expect(
      handleAgenticSetActiveDataset("unknown", manifest, vi.fn())
    ).toEqual({
      status: "error",
      code: "INVALID_DATASET_ID",
      dataset_id: "unknown",
      allowed_dataset_ids: ["iris.csv"],
    });

    expect(
      handleAgenticSetActiveDataset("  ---  ", manifest, vi.fn())
    ).toEqual({
      status: "error",
      code: "INVALID_DATASET_ID",
      dataset_id: "  ---  ",
      allowed_dataset_ids: ["iris.csv"],
    });

    expect(
      handleAgenticSetActiveDataset(
        undefined as unknown as string,
        manifest,
        vi.fn()
      )
    ).toEqual({
      status: "error",
      code: "INVALID_DATASET_ID",
      dataset_id: undefined,
      allowed_dataset_ids: ["iris.csv"],
    });

    expect(
      handleAgenticSetActiveDataset(
        "missing",
        [{ id: "id-only.csv", label: "" }],
        vi.fn()
      )
    ).toEqual({
      status: "error",
      code: "INVALID_DATASET_ID",
      dataset_id: "missing",
      allowed_dataset_ids: ["id-only.csv"],
    });
  });
});
