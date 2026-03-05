/**
 * Chart spec returned by AI tools for deterministic rendering.
 */
export type ChartSpec = {
  id: string;
  title: string;
  description?: string;
  type:
    | "line"
    | "area"
    | "bar"
    | "scatter"
    | "histogram"
    | "density"
    | "roc"
    | "pr"
    | "errorbar"
    | "heatmap"
    | "box"
    | "violin"
    | "biplot"
    | "dendrogram"
    | "surface";
  xKey: string;
  yKeys: string[];
  xLabel?: string;
  yLabel?: string;
  zKey?: string;
  colorKey?: string;
  errorKeys?: Record<string, string>;
  data: Array<Record<string, number | string>>;
  unit?: string;
  currency?: string;
  timeframe?: {
    start: string;
    end: string;
  };
  source?: {
    provider: string;
    url?: string;
  };
  meta?: {
    datasetLabel?: string;
    queryTimeMs?: number;
  };
};
