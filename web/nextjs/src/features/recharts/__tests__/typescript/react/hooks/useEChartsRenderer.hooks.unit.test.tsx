import { act, renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

const { echartsInitMock } = vi.hoisted(() => ({
  echartsInitMock: vi.fn(),
}));
vi.mock("echarts", () => ({
  init: echartsInitMock,
}));

import { useEChartsRenderer } from "@/features/recharts/typescript/react/hooks/useEChartsRenderer.hooks";

describe("useEChartsRenderer", () => {
  it("returns null option for unsupported chart types", () => {
    const { result } = renderHook(() =>
      useEChartsRenderer(
        {
          spec: {
            id: "line",
            title: "Line",
            type: "line",
            xKey: "x",
            yKeys: ["y"],
            data: [{ x: 1, y: 2 }],
          },
        },
        {
          runtime: {
            initChart: vi.fn(),
            bindResize: vi.fn(() => vi.fn()),
          },
        }
      )
    );

    expect(result.current.option).toBeNull();
  });

  it("initializes, sets option, binds resize and disposes on unmount", () => {
    const setOption = vi.fn();
    const resize = vi.fn();
    const dispose = vi.fn();
    const initChart = vi.fn(() => ({ setOption, resize, dispose }));
    const unbind = vi.fn();
    const bindResize = vi.fn(() => unbind);

    const { result, rerender, unmount } = renderHook(() =>
      useEChartsRenderer(
        {
          spec: {
            id: "heat",
            title: "Heat",
            type: "heatmap",
            xKey: "x",
            yKeys: ["y"],
            data: [{ x: "a", y: 1 }],
          },
        },
        {
          runtime: {
            initChart,
            bindResize,
          },
        }
      )
    );

    const div = document.createElement("div");
    act(() => {
      result.current.containerRef.current = div;
    });
    rerender();

    unmount();

    expect(initChart).toHaveBeenCalledWith(div);
    expect(setOption).toHaveBeenCalledTimes(1);
    expect(bindResize).toHaveBeenCalledTimes(1);
    const resizeHandler = bindResize.mock.calls[0]?.[0] as (() => void) | undefined;
    resizeHandler?.();
    expect(resize).toHaveBeenCalledTimes(1);
    expect(unbind).toHaveBeenCalledTimes(1);
    expect(dispose).toHaveBeenCalledTimes(1);
  });

  it("uses default runtime to wire resize listeners", () => {
    const setOption = vi.fn();
    const resize = vi.fn();
    const dispose = vi.fn();
    echartsInitMock.mockReturnValueOnce({ setOption, resize, dispose });

    const addEventListenerSpy = vi.spyOn(window, "addEventListener");
    const removeEventListenerSpy = vi.spyOn(window, "removeEventListener");

    const { result, rerender, unmount } = renderHook(() =>
      useEChartsRenderer({
        spec: {
          id: "heat-default",
          title: "Heat default",
          type: "heatmap",
          xKey: "x",
          yKeys: ["y"],
          data: [{ x: "a", y: 1 }],
        },
      })
    );

    const div = document.createElement("div");
    act(() => {
      result.current.containerRef.current = div;
    });
    rerender();

    const resizeHandler = addEventListenerSpy.mock.calls.find(
      ([eventName]) => eventName === "resize"
    )?.[1] as EventListener | undefined;

    resizeHandler?.(new Event("resize"));
    unmount();

    expect(echartsInitMock).toHaveBeenCalledWith(div);
    expect(setOption).toHaveBeenCalledTimes(1);
    expect(resize).toHaveBeenCalledTimes(1);
    expect(removeEventListenerSpy).toHaveBeenCalledWith("resize", resizeHandler);
    expect(dispose).toHaveBeenCalledTimes(1);

    addEventListenerSpy.mockRestore();
    removeEventListenerSpy.mockRestore();
  });
});
