import { describe, it, expect, vi, beforeEach } from "vitest";
import { ref } from "vue";
import { flushPromises } from "@vue/test-utils";
import { withSetup } from "./helpers/withSetup";
import { useChartRendererOrchestrator } from "~/features/recharts/orchestrator";
import type { ChartSpec } from "~/composables/useChartStore";


const mockSetOption = vi.fn();
const mockDispose = vi.fn();
const mockInit = vi.fn(() => ({ setOption: mockSetOption, dispose: mockDispose }));

vi.mock("echarts", () => ({
  init: (...args: any[]) => mockInit(...args),
}));

describe("useChartRendererOrchestrator", () => {
  beforeEach(() => {
    mockInit.mockClear();
    mockSetOption.mockClear();
    mockDispose.mockClear();
  });

  it("returns a chartEl ref", () => {
    const spec = ref<ChartSpec>({ id: "1", type: "line", title: "Test", data: [{ x: "A", y: 1 }], xKey: "x", yKeys: ["y"] });

    const [result] = withSetup(() => useChartRendererOrchestrator(spec));

    expect(result.chartEl).toBeDefined();
    expect(result.chartEl.value).toBeNull();
  });

  it("initializes echarts on mount when element exists", async () => {
    const spec = ref<ChartSpec>({ id: "1", type: "line", title: "Test", data: [{ x: "A", y: 1 }], xKey: "x", yKeys: ["y"] });

    const [result, app] = withSetup(() => {
      const orchestrator = useChartRendererOrchestrator(spec);
      orchestrator.chartEl.value = document.createElement("div");
      return orchestrator;
    });

    await flushPromises();

    expect(mockInit).toHaveBeenCalledOnce();
    expect(mockSetOption).toHaveBeenCalledOnce();
  });

  it("disposes chart on unmount", async () => {
    const spec = ref<ChartSpec>({ id: "1", type: "line", title: "Test", data: [{ x: "A", y: 1 }], xKey: "x", yKeys: ["y"] });

    const [result, app] = withSetup(() => {
      const orchestrator = useChartRendererOrchestrator(spec);
      orchestrator.chartEl.value = document.createElement("div");
      return orchestrator;
    });

    await flushPromises();
    app.unmount();

    expect(mockDispose).toHaveBeenCalledOnce();
  });
});
