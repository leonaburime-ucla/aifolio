import { useChartStore } from "@/features/recharts/typescript/react/state/zustand/chartStore";

/**
 * AI-facing alias for the shared Recharts store used by external tooling entrypoints.
 */
export const useAiChartStore = useChartStore;
