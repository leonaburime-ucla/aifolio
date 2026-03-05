import { describe, expect, it, vi } from "vitest";
import {
  ensurePytorchTab,
  ensureTensorflowTab,
  waitForPytorchForm,
  waitForTensorflowForm,
} from "@/features/ag-ui-chat/typescript/logic/copilotFrontendToolsFlow.logic";

describe("ensurePytorchTab", () => {
  it("navigates and waits when tab is not pytorch", async () => {
    const setActiveTab = vi.fn();
    const pushRoute = vi.fn();
    const wait = vi.fn(async () => true);

    await ensurePytorchTab({
      activeTab: "charts",
      setActiveTab,
      pushRoute,
      waitForPytorchForm: wait,
    });

    expect(setActiveTab).toHaveBeenCalledWith("pytorch");
    expect(pushRoute).toHaveBeenCalledWith("/ag-ui?page=pytorch");
    expect(wait).toHaveBeenCalledTimes(1);
  });

  it("does not navigate when tab is already pytorch", async () => {
    const setActiveTab = vi.fn();
    const pushRoute = vi.fn();
    const wait = vi.fn(async () => true);

    await ensurePytorchTab({
      activeTab: "pytorch",
      setActiveTab,
      pushRoute,
      waitForPytorchForm: wait,
    });

    expect(setActiveTab).not.toHaveBeenCalled();
    expect(pushRoute).not.toHaveBeenCalled();
    expect(wait).not.toHaveBeenCalled();
  });
});

describe("waitForPytorchForm", () => {
  it("returns true when pytorch form field is present", async () => {
    const result = await waitForPytorchForm(100, {
      querySelector: vi.fn(() => ({ nodeType: 1 } as unknown as Element)),
      delay: async () => {},
    });

    expect(result).toBe(true);
  });

  it("returns false when timeout is zero and field is absent", async () => {
    const result = await waitForPytorchForm(0, {
      querySelector: vi.fn(() => null),
      delay: async () => {},
    });

    expect(result).toBe(false);
  });
});

describe("ensureTensorflowTab", () => {
  it("navigates and waits when tab is not tensorflow", async () => {
    const setActiveTab = vi.fn();
    const pushRoute = vi.fn();
    const wait = vi.fn(async () => true);

    await ensureTensorflowTab({
      activeTab: "charts",
      setActiveTab,
      pushRoute,
      waitForTensorflowForm: wait,
    });

    expect(setActiveTab).toHaveBeenCalledWith("tensorflow");
    expect(pushRoute).toHaveBeenCalledWith("/ag-ui?page=tensorflow");
    expect(wait).toHaveBeenCalledTimes(1);
  });

  it("does not navigate when tab is already tensorflow", async () => {
    const setActiveTab = vi.fn();
    const pushRoute = vi.fn();
    const wait = vi.fn(async () => true);

    await ensureTensorflowTab({
      activeTab: "tensorflow",
      setActiveTab,
      pushRoute,
      waitForTensorflowForm: wait,
    });

    expect(setActiveTab).not.toHaveBeenCalled();
    expect(pushRoute).not.toHaveBeenCalled();
    expect(wait).not.toHaveBeenCalled();
  });
});

describe("waitForTensorflowForm", () => {
  it("returns true when tensorflow form field is present", async () => {
    const result = await waitForTensorflowForm(100, {
      querySelector: vi.fn(() => ({ nodeType: 1 } as unknown as Element)),
      delay: async () => {},
    });

    expect(result).toBe(true);
  });

  it("returns false when timeout is zero and field is absent", async () => {
    const result = await waitForTensorflowForm(0, {
      querySelector: vi.fn(() => null),
      delay: async () => {},
    });

    expect(result).toBe(false);
  });
});
