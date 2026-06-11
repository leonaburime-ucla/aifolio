import { afterEach, describe, expect, it, vi } from "vitest";
import {
  getDefaultTrainingSharedRuntime,
  toClipboardWriteError,
} from "../../../src/ml-training/model/trainingShared";

describe("getDefaultTrainingSharedRuntime", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it("returns runtime with schedule and writeClipboardText", () => {
    const runtime = getDefaultTrainingSharedRuntime();
    expect(typeof runtime.schedule).toBe("function");
    expect(typeof runtime.writeClipboardText).toBe("function");
  });

  it("schedules callbacks through the default runtime", () => {
    vi.useFakeTimers();
    const callback = vi.fn();

    const runtime = getDefaultTrainingSharedRuntime();
    runtime.schedule(callback, 25);
    vi.advanceTimersByTime(24);
    expect(callback).not.toHaveBeenCalled();
    vi.advanceTimersByTime(1);
    expect(callback).toHaveBeenCalledTimes(1);
    vi.useRealTimers();
  });

  it("writes clipboard text through the default browser runtime", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    vi.stubGlobal("navigator", { clipboard: { writeText } });

    const runtime = getDefaultTrainingSharedRuntime();
    await runtime.writeClipboardText("runs");

    expect(writeText).toHaveBeenCalledWith("runs");
  });

  it("rejects clipboard writes when the Clipboard API is unavailable", async () => {
    vi.stubGlobal("navigator", {});

    const runtime = getDefaultTrainingSharedRuntime();

    await expect(runtime.writeClipboardText("runs")).rejects.toThrow("Clipboard API unavailable.");
  });
});

describe("toClipboardWriteError", () => {
  it("extracts message from Error instance", () => {
    const result = toClipboardWriteError({ error: new Error("Permission denied") });
    expect(result).toEqual({
      code: "CLIPBOARD_WRITE_FAILED",
      message: "Permission denied",
    });
  });

  it("returns default message for non-Error", () => {
    const result = toClipboardWriteError({ error: "something" });
    expect(result).toEqual({
      code: "CLIPBOARD_WRITE_FAILED",
      message: "Clipboard write failed.",
    });
  });

  it("returns default message for Error with empty message", () => {
    const result = toClipboardWriteError({ error: new Error("") });
    expect(result).toEqual({
      code: "CLIPBOARD_WRITE_FAILED",
      message: "Clipboard write failed.",
    });
  });
});
