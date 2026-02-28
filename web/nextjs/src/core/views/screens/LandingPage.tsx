import LandingCharts from "@/core/views/screens/LandingCharts";
import LandingChatSidebar from "@/features/ai/views/components/LandingChatSidebar";

export default async function LandingPage() {
  return (
    <div className="flex min-h-screen flex-row bg-zinc-50 text-zinc-900">
      <main className="min-w-0 flex-1 py-10">
        <div className="mx-auto flex max-w-5xl flex-col gap-8 px-6">
          <p className="text-sm font-semibold uppercase tracking-widest text-zinc-500">
            AI-driven Chart Dashboard
          </p>
          <details className="rounded-2xl border border-zinc-200 bg-white/70 p-4 shadow-sm backdrop-blur-sm" open>
            <summary className="cursor-pointer text-sm font-semibold text-zinc-900">
              How to Use Page + Prompts to Try
            </summary>
            <div className="mt-3 space-y-3 text-sm text-zinc-700">
              <p>
                This AI Chat creates charts(Recharts and Echarts) from internal sample data. I do not have APIs for real-time data.
                All data is from the LLM&apos;s internal models.
              </p>
              <div>
                <p className="font-medium text-zinc-900">Sample prompts to try:</p>
                <ul className="mt-2 list-disc space-y-1 pl-5">
                  <li>“Create a line chart of Solana and Bitcoin for the past 5 months.”</li>
                  <li>“Create an area chart of Peruvian beef exports over the past 15 years.”</li>
                  <li>“Show a line chart of Manhattan vs London vs Paris average rent since 2000 as a share of average salary in each of those cities respectively.”</li>
                  <li>“Plot a line chart of US debt levels for the past 50 years. Estimate what it will be for the next 20 in a blue line”</li>
                  <li>“Make a scatter chart comparing Bitcoin and Ethereum returns over the last 30 days.”</li>
                </ul>
              </div>
            </div>
          </details>
          <LandingCharts />
        </div>
      </main>

      <div className="sticky top-16 h-[calc(100vh-64px)] w-[360px] shrink-0 overflow-hidden">
        <LandingChatSidebar />
      </div>
    </div>
  );
}
