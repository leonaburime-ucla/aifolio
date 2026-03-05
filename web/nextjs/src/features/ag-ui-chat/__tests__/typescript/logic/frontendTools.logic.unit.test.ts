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
});
