"use client";

import * as React from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { payments } from "@ui/components/tables/paymentData";

export function ChartCanvas() {
  const chartData = React.useMemo(
    () =>
      payments.map((payment) => ({
        id: payment.id,
        customer: payment.customer,
        amount: payment.amount,
        risk: Number((payment.riskScore * 100).toFixed(2)),
      })),
    [],
  );

  return (
    <section className="w-full space-y-3 rounded-xl border bg-card p-6 shadow-sm">
      <div className="flex flex-col gap-1 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="text-xs uppercase tracking-wide text-muted-foreground">
            Risk vs. Exposure
          </p>
          <h3 className="text-lg font-semibold text-foreground">
            Risk-adjusted payment volume
          </h3>
        </div>
        <span className="text-xs text-muted-foreground">
          plotting risk score (%) against payment amount
        </span>
      </div>
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData} margin={{ top: 10, right: 20, bottom: 0, left: 0 }}>
            <defs>
              <linearGradient id="riskGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="rgb(34,197,94)" stopOpacity={0.8} />
                <stop offset="100%" stopColor="rgb(34,197,94)" stopOpacity={0.05} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--muted-foreground)/0.2)" />
            <XAxis dataKey="customer" tick={{ fontSize: 11 }} />
            <YAxis
              yAxisId="left"
              tickFormatter={(value) => `$${value}`}
              tick={{ fontSize: 11 }}
              axisLine={false}
              width={70}
            />
            <YAxis
              yAxisId="right"
              orientation="right"
              tickFormatter={(value) => `${value}%`}
              tick={{ fontSize: 11 }}
              axisLine={false}
              width={50}
            />
            <Tooltip
              formatter={(value, name) =>
                name === "risk" ? [`${value}%`, "Risk"] : [`$${value}`, "Amount"]
              }
              labelFormatter={(label) => `Customer: ${label}`}
              contentStyle={{ borderRadius: 12 }}
            />
            <Area
              yAxisId="left"
              type="monotone"
              dataKey="amount"
              stroke="rgb(59,130,246)"
              fill="url(#riskGradient)"
              strokeWidth={2}
              name="Amount"
              dot={{ r: 3 }}
            />
            <Area
              yAxisId="right"
              type="linear"
              dataKey="risk"
              stroke="rgb(249,115,22)"
              fill="rgba(249,115,22,0.1)"
              strokeWidth={2}
              name="Risk"
              dot={{ r: 3 }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
