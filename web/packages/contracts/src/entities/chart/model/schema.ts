import { z } from "zod";

export const ChartSpecSchema = z.object({
  id: z.string(),
  title: z.string(),
  description: z.string().optional(),
  type: z.enum([
    "line",
    "area",
    "bar",
    "scatter",
    "histogram",
    "density",
    "roc",
    "pr",
    "errorbar",
    "heatmap",
    "box",
    "violin",
    "biplot",
    "dendrogram",
    "surface",
  ]),
  xKey: z.string(),
  yKeys: z.array(z.string()),
  xLabel: z.string().optional(),
  yLabel: z.string().optional(),
  zKey: z.string().optional(),
  colorKey: z.string().optional(),
  errorKeys: z.record(z.string(), z.string()).optional(),
  data: z.array(z.record(z.string(), z.union([z.number(), z.string()]))),
  unit: z.string().optional(),
  currency: z.string().optional(),
  timeframe: z
    .object({
      start: z.string(),
      end: z.string(),
    })
    .optional(),
  source: z
    .object({
      provider: z.string(),
      url: z.string().optional(),
    })
    .optional(),
  meta: z
    .object({
      datasetLabel: z.string().optional(),
      queryTimeMs: z.number().optional(),
    })
    .optional(),
});
