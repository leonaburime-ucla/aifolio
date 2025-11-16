"use client";

import { ChartCanvas } from "@ui/components/charts/chartCanvas";
import { FabChat } from "@ui/components/chat/FabChat";
import { SamplePaymentsTable } from "@ui/components/tables/sampleTable";
import { SidebarLayout } from "@ui/patterns/sidebarLayout";

const navItems = [
  { label: "Agentic Chat", href: "/" },
  { label: "Tensorflow", href: "/tensorflow" },
  { label: "SciKit Learn", href: "/scikit" },
];

export default function Home() {
  return (
    <SidebarLayout navItems={navItems}>
      <section className="space-y-3 rounded-xl border bg-card p-6 shadow-sm">
        <p className="text-sm uppercase tracking-wide text-muted-foreground">Agentic Chat</p>
        <h1 className="text-2xl font-semibold">Click on the green chat icon to chat about the dataset</h1>
        <p className="text-sm text-muted-foreground">
          Upload or select a dataset, explore it with natural language, and auto-generate charts or
          insights powered by the same pipeline that drives our orchestrated agents.
        </p>
      </section>

      <SamplePaymentsTable />

      <ChartCanvas />

      <FabChat />
    </SidebarLayout>
  );
}
