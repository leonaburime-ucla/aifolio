import { describe, expect, it, vi } from "vitest";
import { renderHook } from "@testing-library/react";

const addChartSpec = vi.fn();
const clearChartSpecs = vi.fn();

vi.mock("@/features/recharts/react/state/zustand/chartStore", () => ({
  useChartStore: (selector: (value: { addChartSpec: typeof addChartSpec; clearChartSpecs: typeof clearChartSpecs }) => unknown) =>
    selector({
      addChartSpec,
      clearChartSpecs,
    }),
}));

import {
  useCopilotChartActionsAdapter,
} from "@/features/recharts/react/ai/state/adapters/chartActions.adapter";
import { useChartManagementAdapter } from "@/features/recharts/react/state/adapters/chartManagement.adapter";

describe("chartActions.adapter", () => {
  it("returns copilot and management action ports", () => {
    const { result: management } = renderHook(() => useChartManagementAdapter());
    const { result: copilot } = renderHook(() => useCopilotChartActionsAdapter());

    management.current.clearChartSpecs();
    copilot.current.clearChartSpecs();

    expect(clearChartSpecs).toHaveBeenCalledTimes(2);
  });
});
