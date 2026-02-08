import { useMemo } from "react";
import { useShallow } from "zustand/react/shallow";
import { useChartStore } from "@/features/recharts/state/zustand/chartStore";
import type { ChartSpec } from "@/features/ai/types/chart.types";

type ChartState = {
  chartSpecs: ChartSpec[];
};

type ChartActions = {
  removeChartSpec: (id: string) => void;
};

export type ChartIntegration = ChartState & ChartActions;

export function useChartOrchestrator(): ChartIntegration {
  const state = useChartStore(
    useShallow((store): ChartState => ({
      chartSpecs: store.chartSpecs,
    }))
  );

  const actions = useMemo<ChartActions>(() => {
    const current = useChartStore.getState();
    return {
      removeChartSpec: current.removeChartSpec,
    };
  }, []);

  return useMemo(
    () => ({
      ...state,
      ...actions,
    }),
    [state, actions]
  );
}
