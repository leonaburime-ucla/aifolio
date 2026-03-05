"use client";

import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  ErrorBar,
  Legend,
  LabelList,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ChartSpec } from "@/features/recharts/__types__/typescript/chart.types";
import {
  formatValue,
  formatXAxisValue,
  SERIES_COLORS,
} from "@/features/recharts/typescript/logic/chartFormatting.logic";
import EChartsRenderer from "@/features/recharts/typescript/react/views/components/EChartsRenderer";
import { Modal } from "@/core/views/components/General/Modal";
import type {
  ChartRendererProps,
  LoadingLabelProps,
} from "@/features/recharts/__types__/typescript/react/views/chartRenderer.types";
import { useChartRendererModel } from "@/features/recharts/typescript/react/hooks/useChartRendererModel.hooks";


/**
 * Draw a lightweight text label for loading points on scatter charts.
 * @param props - Label coordinates and rendered value from Recharts.
 * @returns SVG text node or null when coordinates/value are missing.
 */
export function LoadingLabel({
  x,
  y,
  value,
}: LoadingLabelProps) {
  if (x == null || y == null || value == null) return null;
  return (
    <text x={x + 6} y={y - 6} fontSize={10} fill="#52525b">
      {String(value)}
    </text>
  );
}

/**
 * Render a placeholder for chart types not yet supported in this renderer.
 * @param spec - Chart spec with unsupported type.
 * @returns Placeholder panel.
 */
export function renderUnsupportedChart(spec: ChartSpec) {
  return (
    <div className="rounded-xl border border-dashed border-zinc-200 bg-zinc-50 px-4 py-6 text-sm text-zinc-500">
      <p className="font-semibold text-zinc-700">
        Unsupported chart type: {spec.type}
      </p>
      <p className="mt-2 text-xs">
        This chart type is not yet implemented in the frontend renderer. The
        data is still available in the payload.
      </p>
    </div>
  );
}

export default function ChartRenderer({ spec, onRemove }: ChartRendererProps) {
  const { isExpanded, setIsExpanded, yKeys, scatterLabelKey, chartProps } =
    useChartRendererModel({ spec });

  const commonAxes = (
    <>
      <XAxis
        dataKey={spec.xKey}
        type={spec.type === "scatter" || spec.type === "biplot" ? "number" : "category"}
        tick={{ fontSize: 12 }}
        tickFormatter={(value) => formatXAxisValue({ value, spec })}
        label={spec.xLabel ? { value: spec.xLabel, position: "insideBottom", offset: -10 } : undefined}
      />
      <YAxis
        tick={{ fontSize: 12 }}
        tickFormatter={(value) => formatValue({ value, spec })}
        width={100}
        label={
          spec.yLabel
            ? {
              value: spec.yLabel,
              angle: -90,
              position: "left",
              dx: -8,
              dy: -25,
            }
            : undefined
        }
      />
      <Tooltip
        formatter={(value) => formatValue({ value, spec })}
        content={
          spec.type === "scatter" || spec.type === "biplot"
            ? ({ active, payload: tipPayload }) => {
              if (!active || !tipPayload?.length) return null;
              const entry = tipPayload[0]?.payload as Record<string, unknown> | undefined;
              const featureName = entry?.["feature"] as string | undefined;
              return (
                <div className="rounded-md border border-zinc-200 bg-white px-3 py-2 text-xs shadow-lg">
                  {featureName ? (
                    <p className="mb-1 font-semibold text-zinc-900">{featureName}</p>
                  ) : null}
                  {tipPayload.map((tp) => (
                    <p key={String(tp.dataKey)} className="text-zinc-600">
                      {String(tp.name ?? tp.dataKey)}: {formatValue({ value: tp.value as number, spec })}
                    </p>
                  ))}
                </div>
              );
            }
            : undefined
        }
      />
      {yKeys.length > 1 ? <Legend /> : null}
    </>
  );

  const renderChart = (heightClass: string) => {
    if (
      spec.type === "heatmap" ||
      spec.type === "box" ||
      spec.type === "dendrogram"
    ) {
      return <EChartsRenderer spec={spec} />;
    }

    if (spec.type === "violin" || spec.type === "surface") {
      return renderUnsupportedChart(spec);
    }

    return (
      <div className={`${heightClass} w-full min-w-0`} style={{ minHeight: 260 }}>
        <ResponsiveContainer width="100%" height="100%" minWidth={320} minHeight={240}>
          {spec.type === "area" || spec.type === "density" ? (
            <AreaChart {...chartProps}>
              {commonAxes}
              {yKeys.map((key, index) => (
                <Area
                  key={key}
                  type="monotone"
                  dataKey={key}
                  stroke={SERIES_COLORS[index % SERIES_COLORS.length]}
                  fill={SERIES_COLORS[index % SERIES_COLORS.length]}
                  fillOpacity={0.2}
                />
              ))}
            </AreaChart>
          ) : spec.type === "scatter" || spec.type === "biplot" ? (
            <ScatterChart {...chartProps}>
              {commonAxes}
              {yKeys.map((key, index) => (
                <Scatter
                  key={key}
                  dataKey={key}
                  fill={SERIES_COLORS[index % SERIES_COLORS.length]}
                >
                  {scatterLabelKey ? <LabelList dataKey={scatterLabelKey} content={LoadingLabel} /> : null}
                </Scatter>
              ))}
            </ScatterChart>
          ) : spec.type === "bar" || spec.type === "histogram" ? (
            <BarChart {...chartProps}>
              {commonAxes}
              {yKeys.map((key, index) => (
                <Bar
                  key={key}
                  dataKey={key}
                  fill={SERIES_COLORS[index % SERIES_COLORS.length]}
                  radius={[6, 6, 0, 0]}
                />
              ))}
            </BarChart>
          ) : spec.type === "roc" ||
            spec.type === "pr" ||
            spec.type === "errorbar" ? (
            <LineChart {...chartProps}>
              {commonAxes}
              {yKeys.map((key, index) => (
                <Line
                  key={key}
                  type="monotone"
                  dataKey={key}
                  stroke={SERIES_COLORS[index % SERIES_COLORS.length]}
                  strokeWidth={2}
                  dot={false}
                >
                  {spec.type === "errorbar" && spec.errorKeys?.[key] ? (
                    <ErrorBar dataKey={spec.errorKeys[key]} strokeWidth={1} />
                  ) : null}
                </Line>
              ))}
            </LineChart>
          ) : (
            <LineChart {...chartProps}>
              {commonAxes}
              {yKeys.map((key, index) => (
                <Line
                  key={key}
                  type="monotone"
                  dataKey={key}
                  stroke={SERIES_COLORS[index % SERIES_COLORS.length]}
                  strokeWidth={2}
                  dot={false}
                />
              ))}
            </LineChart>
          )}
        </ResponsiveContainer>
      </div>
    );
  };

  return (
    <>
      <section className="relative mt-3 rounded-2xl border border-white/45 bg-white/55 p-4 shadow-[0_12px_32px_rgba(15,23,42,0.12)] backdrop-blur-md">
        {onRemove ? (
          <button
            type="button"
            onClick={() => onRemove(spec.id)}
            className="absolute -right-2.5 -top-2.5 z-10 flex h-6 w-6 items-center justify-center rounded-full bg-zinc-800 text-xs text-white shadow-md transition-colors hover:bg-red-500"
            aria-label={`Remove chart ${spec.title}`}
          >
            ×
          </button>
        ) : null}
        <div className="mb-4 flex items-start justify-between gap-4">
          <div>
            <h3 className="text-sm font-semibold text-zinc-900">{spec.title}</h3>
            {spec.description ? (
              <p className="text-xs text-zinc-700/80">{spec.description}</p>
            ) : null}
          </div>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => setIsExpanded(true)}
              className="rounded-md border border-white/60 bg-white/60 px-2 py-1 text-xs text-zinc-700 backdrop-blur-sm transition hover:bg-white/80"
              aria-label={`Expand chart ${spec.title}`}
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <polyline points="15 3 21 3 21 9" />
                <polyline points="9 21 3 21 3 15" />
                <line x1="21" y1="3" x2="14" y2="10" />
                <line x1="3" y1="21" x2="10" y2="14" />
              </svg>
            </button>
          </div>
        </div>

        {renderChart("h-72")}
        {spec.meta?.datasetLabel || spec.meta?.queryTimeMs != null ? (
          <div className="mt-3 text-xs text-zinc-700/80">
            {spec.meta?.datasetLabel ? <p>Dataset: {spec.meta.datasetLabel}</p> : null}
            {spec.meta?.queryTimeMs != null ? (
              <p>Query time: {(spec.meta.queryTimeMs / 1000).toFixed(2)}s</p>
            ) : null}
          </div>
        ) : null}
      </section>

      <Modal
        isOpen={isExpanded}
        onClose={() => setIsExpanded(false)}
        title={spec.title}
      >
        <div className="p-4">
          {spec.description ? (
            <p className="mb-6 text-sm text-zinc-500">{spec.description}</p>
          ) : null}
          {renderChart("h-[600px]")}
          {spec.meta?.datasetLabel || spec.meta?.queryTimeMs != null ? (
            <div className="mt-4 text-xs text-zinc-500">
              {spec.meta?.datasetLabel ? <p>Dataset: {spec.meta.datasetLabel}</p> : null}
              {spec.meta?.queryTimeMs != null ? (
                <p>Query time: {(spec.meta.queryTimeMs / 1000).toFixed(2)}s</p>
              ) : null}
            </div>
          ) : null}
        </div>
      </Modal>
    </>
  );
}
