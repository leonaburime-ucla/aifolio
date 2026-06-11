import { describe, expect, it } from "vitest";
import { isAbortError, resolveRuntimeDeps } from "../../../src/chat/model/apiClient";

describe("isAbortError", () => {
  it("returns true for AbortError", () => {
    const error = Object.assign(new Error("aborted"), { name: "AbortError" });
    expect(isAbortError(error)).toBe(true);
  });

  it("returns false for other errors", () => {
    expect(isAbortError(new Error("network"))).toBe(false);
    expect(isAbortError(null)).toBe(false);
    expect(isAbortError(undefined)).toBe(false);
    expect(isAbortError({ name: "TypeError" })).toBe(false);
  });
});

describe("resolveRuntimeDeps", () => {
  it("returns defaults when no deps provided", () => {
    const resolved = resolveRuntimeDeps();
    expect(resolved.debug).toBe(false);
    expect(typeof resolved.fetchImpl).toBe("function");
    expect(typeof resolved.resolveBaseUrl).toBe("function");
    expect(resolved.resolveBaseUrl()).toBe("");
  });

  it("uses provided overrides", () => {
    const customFetch = (() => {}) as unknown as typeof fetch;
    const resolved = resolveRuntimeDeps({
      fetchImpl: customFetch,
      resolveBaseUrl: () => "http://custom",
      debug: true,
    });
    expect(resolved.resolveBaseUrl()).toBe("http://custom");
    expect(resolved.debug).toBe(true);
  });
});
