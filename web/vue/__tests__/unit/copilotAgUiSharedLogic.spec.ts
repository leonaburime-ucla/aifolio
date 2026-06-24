import { describe, expect, it, vi } from "vitest";
import {
  resolveNextAgUiSelectedModelId,
  toReadableModelOptions,
  toReadableDatasetOptions,
  AG_UI_PREFERRED_MODEL_ID,
  AG_UI_FALLBACK_MODELS,
} from "@aifolio/frontend-core/ag-ui";
import {
  toPersistableMessages,
  safeSerialize,
  shouldHydratePersistedMessages,
  shouldSkipEmptyPersistableSync,
} from "@aifolio/frontend-core/ag-ui";
import {
  getAgUiToolsForTab,
  ensureFrameworkTab,
  waitForFrameworkFormField,
} from "@aifolio/frontend-core/ag-ui";

// --- Model Selector ---

describe("resolveNextAgUiSelectedModelId", () => {
  const models = [
    { id: "gemini-3.1-pro-preview", label: "Gemini 3.1 Pro Preview" },
    { id: "m1", label: "Model 1" },
    { id: "m2", label: "Model 2" },
  ];

  it("keeps current selection when still available", () => {
    const result = resolveNextAgUiSelectedModelId({
      currentSelectedModelId: "m2",
      fetchedModels: models,
      apiCurrentModelId: "m1",
      preferredModelId: AG_UI_PREFERRED_MODEL_ID,
    });
    expect(result).toBe("m2");
  });

  it("prefers Gemini 3.1 Pro when no valid current selection", () => {
    const result = resolveNextAgUiSelectedModelId({
      currentSelectedModelId: null,
      fetchedModels: models,
      apiCurrentModelId: "missing",
      preferredModelId: AG_UI_PREFERRED_MODEL_ID,
    });
    expect(result).toBe("gemini-3.1-pro-preview");
  });

  it("falls back to API current model when preferred is unavailable", () => {
    const result = resolveNextAgUiSelectedModelId({
      currentSelectedModelId: null,
      fetchedModels: [{ id: "m1", label: "Model 1" }],
      apiCurrentModelId: "m1",
      preferredModelId: AG_UI_PREFERRED_MODEL_ID,
    });
    expect(result).toBe("m1");
  });

  it("falls back to preferred model when current selection is invalid", () => {
    const result = resolveNextAgUiSelectedModelId({
      currentSelectedModelId: "missing",
      fetchedModels: models,
      apiCurrentModelId: "m1",
      preferredModelId: AG_UI_PREFERRED_MODEL_ID,
    });
    expect(result).toBe("gemini-3.1-pro-preview");
  });

  it("returns null for empty model lists", () => {
    const result = resolveNextAgUiSelectedModelId({
      currentSelectedModelId: "m1",
      fetchedModels: [],
      apiCurrentModelId: "m1",
      preferredModelId: AG_UI_PREFERRED_MODEL_ID,
    });
    expect(result).toBeNull();
  });

  it("falls back to first model when neither preferred nor API model exists", () => {
    const result = resolveNextAgUiSelectedModelId({
      currentSelectedModelId: null,
      fetchedModels: [{ id: "custom", label: "Custom" }],
      apiCurrentModelId: "missing",
      preferredModelId: AG_UI_PREFERRED_MODEL_ID,
    });
    expect(result).toBe("custom");
  });
});

// --- Readable Context ---

describe("toReadableModelOptions", () => {
  it("maps options to id/label pairs", () => {
    const result = toReadableModelOptions([
      { id: "m1", label: "Model 1" },
      { id: "m2", label: "Model 2" },
    ]);
    expect(result).toEqual([
      { id: "m1", label: "Model 1" },
      { id: "m2", label: "Model 2" },
    ]);
  });

  it("returns empty array for empty input", () => {
    expect(toReadableModelOptions([])).toEqual([]);
  });
});

describe("toReadableDatasetOptions", () => {
  it("maps dataset options to id/label pairs", () => {
    const result = toReadableDatasetOptions([
      { id: "d1", label: "Dataset 1" },
      { id: "d2", label: "Dataset 2" },
    ]);
    expect(result).toEqual([
      { id: "d1", label: "Dataset 1" },
      { id: "d2", label: "Dataset 2" },
    ]);
  });

  it("returns empty array for empty input", () => {
    expect(toReadableDatasetOptions([])).toEqual([]);
  });
});

// --- AG-UI Model Defaults ---

describe("AG_UI_FALLBACK_MODELS", () => {
  it("includes Gemini 3.1 Pro Preview as the first fallback model", () => {
    expect(AG_UI_FALLBACK_MODELS[0].id).toBe("gemini-3.1-pro-preview");
  });

  it("has at least 2 fallback models", () => {
    expect(AG_UI_FALLBACK_MODELS.length).toBeGreaterThanOrEqual(2);
  });
});

// --- Message Persistence ---

describe("toPersistableMessages", () => {
  it("normalizes valid messages", () => {
    const messages = [
      { id: "u1", type: "TextMessage", role: "user", content: "hello" },
      { id: "a1", type: "TextMessage", role: "assistant", content: "world" },
    ];
    const result = toPersistableMessages(messages);
    expect(result.length).toBe(2);
    expect(result[0]).toEqual({ id: "u1", type: "TextMessage", role: "user", content: "hello" });
  });

  it("filters out coagent-state-render messages", () => {
    const messages = [
      { id: "u1", role: "user", content: "hello" },
      { id: "coagent-state-render-agentic-research", role: "assistant", content: "" },
    ];
    const result = toPersistableMessages(messages);
    expect(result.length).toBe(1);
    expect(result[0]).toMatchObject({ id: "u1" });
  });

  it("deduplicates by id (keeps last)", () => {
    const messages = [
      { id: "u1", role: "user", content: "first" },
      { id: "u1", role: "user", content: "second" },
    ];
    const result = toPersistableMessages(messages);
    expect(result.length).toBe(1);
    expect((result[0] as any).content).toBe("second");
  });

  it("concatenates array content parts", () => {
    const messages = [
      { id: "a1", role: "assistant", content: ["hello", { text: "world" }, { content: "again" }] },
    ];
    const result = toPersistableMessages(messages);
    expect(result.length).toBe(1);
    expect((result[0] as any).content).toBe("hello\nworld\nagain");
  });

  it("filters out messages with empty content", () => {
    const messages = [
      { id: "u1", role: "user", content: "" },
      { id: "u2", role: "user", content: "   " },
    ];
    expect(toPersistableMessages(messages).length).toBe(0);
  });

  it("strips functions and symbols from nested objects", () => {
    const messages = [
      { id: "u1", role: "user", content: "hello", callback: () => {} },
    ];
    const result = toPersistableMessages(messages);
    expect(result.length).toBe(1);
    expect((result[0] as any).callback).toBeUndefined();
  });

  it("returns empty array for non-serializable input", () => {
    const circular: any = { id: "u1", role: "user", content: "hello" };
    circular.self = circular;
    expect(toPersistableMessages([circular]).length).toBe(0);
  });
});

describe("safeSerialize", () => {
  it("serializes normal objects", () => {
    expect(safeSerialize({ key: "value" })).toBe('{"key":"value"}');
  });

  it("returns empty string for circular references", () => {
    const circular: any = {};
    circular.self = circular;
    expect(safeSerialize(circular)).toBe("");
  });
});

describe("shouldHydratePersistedMessages", () => {
  it("returns true when live is empty and persisted exists", () => {
    expect(shouldHydratePersistedMessages({
      livePersistableCount: 0,
      liveUserMessageCount: 0,
      persistedCount: 5,
    })).toBe(true);
  });

  it("returns false when persisted is empty", () => {
    expect(shouldHydratePersistedMessages({
      livePersistableCount: 0,
      liveUserMessageCount: 0,
      persistedCount: 0,
    })).toBe(false);
  });

  it("returns false when live already has user messages", () => {
    expect(shouldHydratePersistedMessages({
      livePersistableCount: 2,
      liveUserMessageCount: 1,
      persistedCount: 5,
    })).toBe(false);
  });

  it("returns true when live has no user messages but no persistable content", () => {
    expect(shouldHydratePersistedMessages({
      livePersistableCount: 0,
      liveUserMessageCount: 0,
      persistedCount: 3,
    })).toBe(true);
  });
});

describe("shouldSkipEmptyPersistableSync", () => {
  it("returns true when live is empty but persisted has data", () => {
    expect(shouldSkipEmptyPersistableSync({
      livePersistableCount: 0,
      persistedCount: 5,
    })).toBe(true);
  });

  it("returns false when live has persistable content", () => {
    expect(shouldSkipEmptyPersistableSync({
      livePersistableCount: 3,
      persistedCount: 5,
    })).toBe(false);
  });

  it("returns false when both are empty", () => {
    expect(shouldSkipEmptyPersistableSync({
      livePersistableCount: 0,
      persistedCount: 0,
    })).toBe(false);
  });
});

// --- Tools Catalog ---

describe("getAgUiToolsForTab", () => {
  it("includes switch_ag_ui_tab in all tabs", () => {
    const tabs = ["charts", "agentic-research", "pytorch", "tensorflow"] as const;
    for (const tab of tabs) {
      const tools = getAgUiToolsForTab(tab);
      expect(tools.some((t) => t.name === "switch_ag_ui_tab")).toBe(true);
    }
  });

  it("includes chart tools for charts tab", () => {
    const tools = getAgUiToolsForTab("charts");
    expect(tools.some((t) => t.name === "add_chart_spec")).toBe(true);
    expect(tools.some((t) => t.name === "clear_charts")).toBe(true);
  });

  it("includes agentic research tools for agentic-research tab", () => {
    const tools = getAgUiToolsForTab("agentic-research");
    expect(tools.some((t) => t.name === "ar-add_chart_spec")).toBe(true);
    expect(tools.some((t) => t.name === "ar-set_active_dataset")).toBe(true);
  });

  it("includes pytorch tools for pytorch tab", () => {
    const tools = getAgUiToolsForTab("pytorch");
    expect(tools.some((t) => t.name === "set_pytorch_form_fields")).toBe(true);
    expect(tools.some((t) => t.name === "start_pytorch_training_runs")).toBe(true);
    expect(tools.some((t) => t.name === "train_pytorch_model")).toBe(true);
    expect(tools.some((t) => t.name === "set_active_ml_form_fields")).toBe(true);
  });

  it("includes tensorflow tools for tensorflow tab", () => {
    const tools = getAgUiToolsForTab("tensorflow");
    expect(tools.some((t) => t.name === "set_tensorflow_form_fields")).toBe(true);
    expect(tools.some((t) => t.name === "start_tensorflow_training_runs")).toBe(true);
    expect(tools.some((t) => t.name === "train_tensorflow_model")).toBe(true);
  });

  it("does not include pytorch-specific tools in tensorflow tab", () => {
    const tools = getAgUiToolsForTab("tensorflow");
    expect(tools.some((t) => t.name === "set_pytorch_form_fields")).toBe(false);
    expect(tools.some((t) => t.name === "start_pytorch_training_runs")).toBe(false);
  });

  it("does not include tensorflow-specific tools in pytorch tab", () => {
    const tools = getAgUiToolsForTab("pytorch");
    expect(tools.some((t) => t.name === "set_tensorflow_form_fields")).toBe(false);
    expect(tools.some((t) => t.name === "start_tensorflow_training_runs")).toBe(false);
  });
});

// --- ML Tools Flow ---

describe("ensureFrameworkTab", () => {
  it("navigates and waits when tab is not the target framework", async () => {
    const setActiveTab = vi.fn();
    const pushRoute = vi.fn();
    const waitForFrameworkForm = vi.fn(async () => {});

    await ensureFrameworkTab({
      activeTab: "charts",
      setActiveTab,
      pushRoute,
      frameworkTab: "pytorch",
      waitForFrameworkForm,
    });

    expect(setActiveTab).toHaveBeenCalledWith("pytorch");
    expect(pushRoute).toHaveBeenCalledWith("/ag-ui?page=pytorch");
    expect(waitForFrameworkForm).toHaveBeenCalledTimes(1);
  });

  it("does not navigate when already on the target tab", async () => {
    const setActiveTab = vi.fn();
    const pushRoute = vi.fn();
    const waitForFrameworkForm = vi.fn(async () => {});

    await ensureFrameworkTab({
      activeTab: "pytorch",
      setActiveTab,
      pushRoute,
      frameworkTab: "pytorch",
      waitForFrameworkForm,
    });

    expect(setActiveTab).not.toHaveBeenCalled();
    expect(pushRoute).not.toHaveBeenCalled();
    expect(waitForFrameworkForm).toHaveBeenCalledTimes(1);
  });

  it("works for tensorflow target", async () => {
    const setActiveTab = vi.fn();
    const pushRoute = vi.fn();
    const waitForFrameworkForm = vi.fn(async () => {});

    await ensureFrameworkTab({
      activeTab: "charts",
      setActiveTab,
      pushRoute,
      frameworkTab: "tensorflow",
      waitForFrameworkForm,
    });

    expect(setActiveTab).toHaveBeenCalledWith("tensorflow");
    expect(pushRoute).toHaveBeenCalledWith("/ag-ui?page=tensorflow");
  });
});

describe("waitForFrameworkFormField", () => {
  it("returns true immediately when element is present", async () => {
    const result = await waitForFrameworkFormField("[data-testid='form']", 100, {
      querySelector: vi.fn(() => ({ nodeType: 1 } as unknown as Element)),
      delay: async () => {},
    });
    expect(result).toBe(true);
  });

  it("returns false when element never appears and timeout expires", async () => {
    const result = await waitForFrameworkFormField("[data-testid='form']", 200, {
      querySelector: vi.fn(() => null),
      delay: async () => {},
    });
    expect(result).toBe(false);
  });

  it("returns true when element appears after retries", async () => {
    let callCount = 0;
    const result = await waitForFrameworkFormField("[data-testid='form']", 5000, {
      querySelector: vi.fn(() => {
        callCount++;
        return callCount >= 3 ? ({ nodeType: 1 } as unknown as Element) : null;
      }),
      delay: async () => {},
    });
    expect(result).toBe(true);
  });
});
