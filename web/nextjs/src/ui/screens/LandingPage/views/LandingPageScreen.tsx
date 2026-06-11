"use client";

import { Suspense, useEffect } from "react";
import ChartsWorkspaceSurface from "@/features/recharts/react/views/screens/ChartsWorkspaceSurface";
import dynamic from "next/dynamic";
import UIFeedback from "@/ui/screens/LandingPage/views/components/UIFeedback";

const DEBUG_EFFECTS = process.env.NEXT_PUBLIC_DEBUG_EFFECTS === "1";
const DEFAULT_DEMO_TOAST_DURATION_MS = 4000;

function getDebugPath(): string {
  return globalThis.location?.pathname ?? "";
}

const LandingChatSidebar = dynamic(
  // Prevent SSR for chat sidebar because nested chat hooks touch `window`.
  // Without this, prerendering `/chat` fails on the server.
  () => import("@/ui/screens/LandingPage/chat/views/LandingChatSidebar"),
  { ssr: false }
);

type LandingPageScreenProps = {
  showSidebar?: boolean;
  showTitle?: boolean;
  horizontalPadding?: boolean;
};

export default function LandingPageScreen({
  showSidebar = true,
  showTitle = true,
  horizontalPadding = true,
}: LandingPageScreenProps) {
  useEffect(() => {
    if (DEBUG_EFFECTS) {
      console.log("[page-debug] landing_page_mounted", {
        path: getDebugPath(),
        showSidebar,
        showTitle,
        horizontalPadding,
      });
    }
  }, [horizontalPadding, showSidebar, showTitle]);

  return (
    <div className="flex min-h-screen flex-row bg-zinc-50 text-zinc-900">
      <Suspense fallback={null}>
        <UIFeedback />
      </Suspense>
      <main className="min-w-0 flex-1 py-10">
        <div
          className={`mx-auto flex max-w-5xl flex-col gap-8 ${
            horizontalPadding ? "px-6" : "px-0"
          }`}
        >
          {showTitle ? (
            <p className="text-sm font-semibold uppercase tracking-widest text-zinc-500">
              AI-driven Chart Dashboard
            </p>
          ) : null}
          <ChartsWorkspaceSurface />
        </div>
      </main>

      {showSidebar ? (
        <div className="sticky top-16 h-[calc(100vh-64px)] w-[360px] shrink-0 overflow-hidden">
          <LandingChatSidebar />
        </div>
      ) : null}
    </div>
  );
}
