import type { ChartSpec } from "@/features/ai/types/chart.types";

export type ChatApiResponse = {
  status: "ok" | "error";
  result?:
    | string
    | Array<{ type: string; text?: string }>
    | { message?: string; chartSpec?: ChartSpec | ChartSpec[] | null };
  error?: string;
  model?: string;
};

export type ModelsApiResponse = {
  status: "ok" | "error";
  currentModel?: string;
  models?: Array<{ id: string; label: string }>;
  error?: string;
};
