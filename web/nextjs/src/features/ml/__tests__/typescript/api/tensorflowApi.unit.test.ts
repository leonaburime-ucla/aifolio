import { describe, expect, it, vi } from "vitest";
import {
  distillTensorflowModel,
  trainTensorflowModel,
} from "@/features/ml/typescript/api/tensorflowApi";

describe("tensorflowApi", () => {
  it("returns ok payload for successful train response", async () => {
    const fetchImpl = vi.fn(async () => ({
      ok: true,
      json: async () => ({
        status: "ok",
        run_id: "run-1",
        model_id: "model-1",
      }),
    }));

    const result = await trainTensorflowModel(
      { dataset_id: "d1.csv", target_column: "target" },
      { fetchImpl: fetchImpl as unknown as typeof fetch }
    );

    expect(result.status).toBe("ok");
  });

  it("returns error payload for unsuccessful train response", async () => {
    const fetchImpl = vi.fn(async () => ({
      ok: false,
      json: async () => ({ status: "error", error: "bad train" }),
    }));

    const result = await trainTensorflowModel(
      { dataset_id: "d1.csv", target_column: "target" },
      { fetchImpl: fetchImpl as unknown as typeof fetch }
    );

    expect(result).toEqual({
      status: "error",
      code: "TENSORFLOW_TRAIN_FAILED",
      error: "bad train",
    });
  });

  it("returns request failure when train throws", async () => {
    const fetchImpl = vi.fn(async () => {
      throw new Error("network");
    });

    const result = await trainTensorflowModel(
      { dataset_id: "d1.csv", target_column: "target" },
      { fetchImpl: fetchImpl as unknown as typeof fetch }
    );

    expect(result.status).toBe("error");
    expect(result.code).toBe("TENSORFLOW_TRAIN_REQUEST_FAILED");
  });

  it("uses default train failure messages for non-ok status and non-Error throws", async () => {
    const nonOk = await trainTensorflowModel(
      { dataset_id: "d1.csv", target_column: "target" },
      {
        fetchImpl: vi.fn(async () => ({
          ok: true,
          json: async () => ({ status: "error" }),
        })) as unknown as typeof fetch,
      }
    );
    expect(nonOk).toEqual({
      status: "error",
      code: "TENSORFLOW_TRAIN_FAILED",
      error: "Failed to train TensorFlow model.",
    });

    const thrown = await trainTensorflowModel(
      { dataset_id: "d1.csv", target_column: "target" },
      {
        fetchImpl: vi.fn(async () => {
          throw "network";
        }) as unknown as typeof fetch,
      }
    );
    expect(thrown).toEqual({
      status: "error",
      code: "TENSORFLOW_TRAIN_REQUEST_FAILED",
      error: "Failed to send TensorFlow training request.",
    });
  });

  it("returns ok payload for successful distill response and clears timeout", async () => {
    const fetchImpl = vi.fn(async () => ({
      ok: true,
      json: async () => ({ status: "ok", run_id: "distill-1" }),
    }));
    const scheduleTimeout = vi.fn((cb: () => void) => {
      void cb;
      return 123 as unknown as ReturnType<typeof setTimeout>;
    });
    const clearScheduledTimeout = vi.fn();

    const result = await distillTensorflowModel(
      { dataset_id: "d1.csv", target_column: "target" },
      {
        fetchImpl: fetchImpl as unknown as typeof fetch,
        scheduleTimeout,
        clearScheduledTimeout,
      }
    );

    expect(result.status).toBe("ok");
    expect(scheduleTimeout).toHaveBeenCalledTimes(1);
    expect(clearScheduledTimeout).toHaveBeenCalledWith(123);
  });

  it("returns timeout error when distill throws AbortError", async () => {
    const abortError = new Error("aborted");
    abortError.name = "AbortError";
    const fetchImpl = vi.fn(async () => {
      throw abortError;
    });
    const scheduleTimeout = vi.fn(() => 999 as unknown as ReturnType<typeof setTimeout>);
    const clearScheduledTimeout = vi.fn();

    const result = await distillTensorflowModel(
      { dataset_id: "d1.csv", target_column: "target" },
      {
        fetchImpl: fetchImpl as unknown as typeof fetch,
        scheduleTimeout,
        clearScheduledTimeout,
      }
    );

    expect(result).toEqual({
      status: "error",
      code: "TENSORFLOW_DISTILL_REQUEST_FAILED",
      error: "Distillation timed out after 180 seconds.",
    });
    expect(clearScheduledTimeout).toHaveBeenCalledWith(999);
  });

  it("uses default distill failure messages for response and non-Error throws", async () => {
    const failed = await distillTensorflowModel(
      { dataset_id: "d1.csv", target_column: "target" },
      {
        fetchImpl: vi.fn(async () => ({
          ok: true,
          json: async () => ({ status: "error" }),
        })) as unknown as typeof fetch,
        scheduleTimeout: vi.fn(() => 1 as unknown as ReturnType<typeof setTimeout>),
        clearScheduledTimeout: vi.fn(),
      }
    );
    expect(failed).toEqual({
      status: "error",
      code: "TENSORFLOW_DISTILL_FAILED",
      error: "Failed to distill TensorFlow model.",
    });

    const thrown = await distillTensorflowModel(
      { dataset_id: "d1.csv", target_column: "target" },
      {
        fetchImpl: vi.fn(async () => {
          throw "network";
        }) as unknown as typeof fetch,
        scheduleTimeout: vi.fn(() => 2 as unknown as ReturnType<typeof setTimeout>),
        clearScheduledTimeout: vi.fn(),
      }
    );
    expect(thrown).toEqual({
      status: "error",
      code: "TENSORFLOW_DISTILL_REQUEST_FAILED",
      error: "Failed to send TensorFlow distillation request.",
    });
  });
});
