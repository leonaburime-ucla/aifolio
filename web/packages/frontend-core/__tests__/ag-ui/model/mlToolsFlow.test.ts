import { afterEach, describe, expect, it, vi } from "vitest";
import {
  waitForFrameworkFormField,
  ensureFrameworkTab,
} from "../../../src/ag-ui/model/mlToolsFlow";

describe("waitForFrameworkFormField", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("returns true immediately when element exists", async () => {
    const fakeElement = { tagName: "DIV" };
    const querySelector = vi.fn().mockReturnValue(fakeElement);
    const result = await waitForFrameworkFormField("#form", 500, { querySelector });
    expect(result).toBe(true);
    expect(querySelector).toHaveBeenCalledWith("#form");
  });

  it("returns false after timeout when element never appears", async () => {
    const querySelector = vi.fn().mockReturnValue(null);
    const delay = vi.fn().mockResolvedValue(undefined);
    const result = await waitForFrameworkFormField("#missing", 50, {
      querySelector,
      delay,
    });
    expect(result).toBe(false);
  });

  it("retries until element appears", async () => {
    let calls = 0;
    const fakeElement = { tagName: "INPUT" };
    const querySelector = vi.fn(() => {
      calls++;
      return calls >= 3 ? fakeElement : null;
    });
    const delay = vi.fn().mockResolvedValue(undefined);
    const result = await waitForFrameworkFormField("#field", 5000, {
      querySelector,
      delay,
    });
    expect(result).toBe(true);
    expect(calls).toBeGreaterThanOrEqual(3);
  });

  it("uses the default document selector and delay when runtime overrides are omitted", async () => {
    const querySelector = vi.fn().mockReturnValue(null);
    vi.stubGlobal("document", { querySelector });

    const result = await waitForFrameworkFormField("#missing", 1);

    expect(result).toBe(false);
    expect(querySelector).toHaveBeenCalledWith("#missing");
  });
});

describe("ensureFrameworkTab", () => {
  it("switches tab and pushes route when not on target tab", async () => {
    const setActiveTab = vi.fn();
    const pushRoute = vi.fn();
    const waitForFrameworkForm = vi.fn().mockResolvedValue(undefined);

    await ensureFrameworkTab({
      activeTab: "pytorch",
      setActiveTab,
      pushRoute,
      frameworkTab: "tensorflow",
      waitForFrameworkForm,
    });

    expect(setActiveTab).toHaveBeenCalledWith("tensorflow");
    expect(pushRoute).toHaveBeenCalledWith("/ag-ui?page=tensorflow");
    expect(waitForFrameworkForm).toHaveBeenCalled();
  });

  it("does not switch tab when already on target", async () => {
    const setActiveTab = vi.fn();
    const pushRoute = vi.fn();
    const waitForFrameworkForm = vi.fn().mockResolvedValue(undefined);

    await ensureFrameworkTab({
      activeTab: "pytorch",
      setActiveTab,
      pushRoute,
      frameworkTab: "pytorch",
      waitForFrameworkForm,
    });

    expect(setActiveTab).not.toHaveBeenCalled();
    expect(pushRoute).not.toHaveBeenCalled();
    expect(waitForFrameworkForm).toHaveBeenCalled();
  });
});
