import { readFile } from "fs/promises";
import path from "path";
import ChatSidebar from "@/features/ai/views/components/ChatSidebar";
import LandingCharts from "@/core/views/screens/LandingCharts";

async function loadJson<T>(fileName: string): Promise<T> {
  const filePath = path.resolve(process.cwd(), "..", "sample-data", fileName);
  const raw = await readFile(filePath, "utf-8");
  return JSON.parse(raw) as T;
}

export default async function LandingPage() {
  const [coin, chart] = await Promise.all([
    loadJson("coin_bitcoin.json"),
    loadJson("market-chart_bitcoin_30d_usd.json"),
  ]);

  return (
    <div className="flex min-h-screen flex-row bg-zinc-50 text-zinc-900">
      <main className="min-w-0 flex-1 py-10">
        <div className="mx-auto flex max-w-5xl flex-col gap-8 px-6">
          <p className="text-sm font-semibold uppercase tracking-widest text-zinc-500">
            AI-driven crypto dashboard
          </p>
          <LandingCharts coin={coin} chart={chart} />
        </div>
      </main>

      <div className="sticky top-0 h-screen w-[360px] shrink-0 overflow-hidden">
        <ChatSidebar />
      </div>
    </div>
  );
}
