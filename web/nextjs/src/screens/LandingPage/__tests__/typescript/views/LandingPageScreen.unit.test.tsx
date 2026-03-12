import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { toastMock, useSearchParamsMock } = vi.hoisted(() => {
  const mock = vi.fn();
  mock.custom = vi.fn();
  mock.error = vi.fn();
  mock.success = vi.fn();
  mock.dismiss = vi.fn();

  return {
    toastMock: mock,
  useSearchParamsMock: vi.fn(),
  };
});

vi.mock("next/dynamic", () => ({
  default: () => () => <div data-testid="landing-chat-sidebar" />,
}));

vi.mock("next/navigation", () => ({
  useSearchParams: useSearchParamsMock,
}));

vi.mock("react-hot-toast", () => ({
  toast: toastMock,
}));

vi.mock("@/core/views/screens/LandingCharts", () => ({
  default: () => <div data-testid="landing-charts" />,
}));

import LandingPageScreen from "@/screens/LandingPage/views/LandingPageScreen";

function setDemoToastParams(input: {
  kind: string | null;
  dismiss?: string | null;
  duration?: string | null;
}): void {
  useSearchParamsMock.mockReturnValue({
    get: (key: string) => {
      if (key === "demo-toast") return input.kind;
      if (key === "demo-toast-dismiss") return input.dismiss ?? null;
      if (key === "demo-toast-duration") return input.duration ?? null;
      return null;
    },
  });
}

describe("LandingPageScreen", () => {
  beforeEach(() => {
    toastMock.mockClear();
    toastMock.custom.mockClear();
    toastMock.error.mockClear();
    toastMock.success.mockClear();
    toastMock.dismiss.mockClear();
  });

  afterEach(() => {
    cleanup();
  });

  it("renders the page chrome and fires an error demo toast from the URL", () => {
    setDemoToastParams({ kind: "error" });

    render(<LandingPageScreen />);

    expect(screen.getByTestId("landing-charts")).toBeInTheDocument();
    expect(toastMock.custom).toHaveBeenCalledWith(
      expect.any(Function),
      expect.objectContaining({
        id: "landing-page-demo-toast",
        position: "top-center",
        duration: 4000,
      })
    );
  });

  it("fires warning and success demo toasts from the URL", () => {
    setDemoToastParams({ kind: "warning" });
    const { rerender } = render(<LandingPageScreen />);

    expect(toastMock.custom).toHaveBeenCalledWith(
      expect.any(Function),
      expect.objectContaining({
        id: "landing-page-demo-toast",
        position: "top-center",
        duration: 4000,
      })
    );

    setDemoToastParams({ kind: "success" });
    rerender(<LandingPageScreen />);

    expect(toastMock.custom).toHaveBeenLastCalledWith(
      expect.any(Function),
      expect.objectContaining({
        id: "landing-page-demo-toast",
        position: "top-center",
        duration: 4000,
      })
    );
  });

  it("uses a dismiss button mode when requested by the URL", () => {
    setDemoToastParams({ kind: "warning", dismiss: "1" });

    render(<LandingPageScreen />);

    expect(toastMock.custom).toHaveBeenCalledWith(
      expect.any(Function),
      expect.objectContaining({
        id: "landing-page-demo-toast",
        position: "top-center",
        duration: Infinity,
      })
    );
  });

  it("dismisses the demo toast when the URL flag is removed", () => {
    setDemoToastParams({ kind: "error" });
    const { rerender } = render(<LandingPageScreen />);

    setDemoToastParams({ kind: null });
    rerender(<LandingPageScreen />);

    expect(toastMock.dismiss).toHaveBeenCalledWith("landing-page-demo-toast");
  });
});
