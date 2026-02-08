"use client";

import { useState } from "react";
import {
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type CoinMarketData = {
  market_data?: {
    current_price?: { usd?: number };
    market_cap?: { usd?: number };
    total_volume?: { usd?: number };
  };
  name?: string;
  symbol?: string;
};

type MarketChart = {
  prices?: Array<[number, number]>;
  market_caps?: Array<[number, number]>;
  total_volumes?: Array<[number, number]>;
};

type PricePoint = {
  date: string;
  price: number;
};

type BitcoinChartsProps = {
  coin: CoinMarketData;
  chart: MarketChart;
};

function formatCurrency(value?: number) {
  if (value === undefined) return "—";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(value);
}

export default function BitcoinCharts({ coin, chart }: BitcoinChartsProps) {
  const [isVisible, setIsVisible] = useState(true);

  if (!isVisible) return null;

  const priceSeries: PricePoint[] = (chart.prices ?? []).map(([time, price]) => ({
    date: new Date(time).toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
    }),
    price,
  }));

  return (
    <section className="relative rounded-3xl border border-zinc-200 bg-white p-6 shadow-sm">
      <button
        type="button"
        onClick={() => setIsVisible(false)}
        aria-label="Remove chart"
        className="absolute -right-3 -top-3 z-10 flex h-8 w-8 items-center justify-center rounded-full border border-zinc-200 bg-white text-zinc-500 shadow-sm transition hover:bg-zinc-50"
      >
        ×
      </button>
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <p className="text-sm font-semibold uppercase tracking-widest text-zinc-500">
            {coin.name ?? "Bitcoin"}
          </p>
          <h2 className="text-2xl font-semibold text-zinc-900">30D Price</h2>
        </div>
        <div className="flex flex-wrap gap-6 text-sm text-zinc-600">
          <div>
            <p className="text-xs uppercase tracking-wide text-zinc-400">Price</p>
            <p className="font-semibold text-zinc-900">
              {formatCurrency(coin.market_data?.current_price?.usd)}
            </p>
          </div>
          <div>
            <p className="text-xs uppercase tracking-wide text-zinc-400">
              Market Cap
            </p>
            <p className="font-semibold text-zinc-900">
              {formatCurrency(coin.market_data?.market_cap?.usd)}
            </p>
          </div>
          <div>
            <p className="text-xs uppercase tracking-wide text-zinc-400">
              24h Volume
            </p>
            <p className="font-semibold text-zinc-900">
              {formatCurrency(coin.market_data?.total_volume?.usd)}
            </p>
          </div>
        </div>
      </div>

      <div className="mt-6 h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={priceSeries}
            margin={{ top: 10, right: 16, left: 0, bottom: 0 }}
          >
            <XAxis dataKey="date" tick={{ fontSize: 12 }} />
            <YAxis
              tickFormatter={(value) => `$${Math.round(value / 1000)}k`}
              width={48}
              tick={{ fontSize: 12 }}
            />
            <Tooltip
              formatter={(value: any) => formatCurrency(value)}
              labelClassName="text-xs"
            />
            <Line
              type="monotone"
              dataKey="price"
              stroke="#18181b"
              strokeWidth={2}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
