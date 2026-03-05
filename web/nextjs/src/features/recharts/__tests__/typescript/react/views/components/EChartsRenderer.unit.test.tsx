import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";

const useEChartsRendererMock = vi.fn();

vi.mock("@/features/recharts/typescript/react/hooks/useEChartsRenderer.hooks", () => ({
  useEChartsRenderer: (params: unknown) => useEChartsRendererMock(params),
}));

import EChartsRenderer from "@/features/recharts/typescript/react/views/components/EChartsRenderer";

describe("EChartsRenderer", () => {
  it("renders nothing when option is null", () => {
    useEChartsRendererMock.mockReturnValueOnce({ containerRef: { current: null }, option: null });

    const { container } = render(
      <EChartsRenderer
        spec={{ id: "c1", title: "C", type: "heatmap", xKey: "x", yKeys: ["y"], data: [] }}
      />
    );

    expect(container.firstChild).toBeNull();
  });

  it("renders container when option exists", () => {
    useEChartsRendererMock.mockReturnValueOnce({
      containerRef: { current: null },
      option: { series: [] },
    });

    const { container } = render(
      <EChartsRenderer
        spec={{ id: "c1", title: "C", type: "heatmap", xKey: "x", yKeys: ["y"], data: [] }}
      />
    );

    expect(container.querySelector(".h-64.w-full")).toBeTruthy();
    expect(screen.queryByText("C")).toBeNull();
  });
});
