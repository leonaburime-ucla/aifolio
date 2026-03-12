"use client";

import { useEffect, useRef } from "react";
import { useSearchParams } from "next/navigation";
import { toast, type Toast } from "react-hot-toast";

const LANDING_PAGE_DEMO_TOAST_ID = "landing-page-demo-toast";
const DEFAULT_DEMO_TOAST_DURATION_MS = 4000;

type DemoToastKind = "error" | "warning" | "success";

type DemoToastConfig = {
  dismissible: boolean;
  durationMs: number;
};

type DemoToastVisuals = {
  title: string;
  message: string;
  accentClassName: string;
  icon: string;
};

function resolveDemoToastConfig(searchParams: URLSearchParams | null): {
  kind: DemoToastKind | null;
  config: DemoToastConfig;
  signature: string;
} {
  const rawKind = searchParams?.get("demo-toast");
  const rawDismiss = searchParams?.get("demo-toast-dismiss");
  const rawDuration = searchParams?.get("demo-toast-duration");
  const parsedDuration = Number(rawDuration);
  const durationMs =
    Number.isFinite(parsedDuration) && parsedDuration > 0
      ? parsedDuration
      : DEFAULT_DEMO_TOAST_DURATION_MS;
  const dismissible = rawDismiss === "1" || rawDismiss === "true";
  const kind: DemoToastKind | null =
    rawKind === "error" || rawKind === "warning" || rawKind === "success"
      ? rawKind
      : null;

  return {
    kind,
    config: {
      dismissible,
      durationMs,
    },
    signature: [rawKind ?? "", dismissible ? "dismiss" : "auto", durationMs].join(":"),
  };
}

function getDemoToastVisuals(kind: DemoToastKind): DemoToastVisuals {
  if (kind === "error") {
    return {
      title: "Error",
      message: "Demo error: page-level error toast is working.",
      accentClassName: "border-red-200 bg-red-50 text-red-700",
      icon: "!",
    };
  }

  if (kind === "warning") {
    return {
      title: "Warning",
      message: "Demo warning: page-level warning toast is working.",
      accentClassName: "border-amber-200 bg-amber-50 text-amber-800",
      icon: "!",
    };
  }

  return {
    title: "Success",
    message: "Demo success: page-level success toast is working.",
    accentClassName: "border-emerald-200 bg-emerald-50 text-emerald-700",
    icon: "✓",
  };
}

function showDemoToast(kind: DemoToastKind, config: DemoToastConfig): void {
  const visuals = getDemoToastVisuals(kind);

  toast.custom(
    (toastInstance: Toast) => (
      <div
        className={`pointer-events-auto flex w-[360px] gap-3 rounded-lg border px-4 py-3 shadow-lg ${visuals.accentClassName}`}
      >
        <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full border border-current/20 bg-white/70 text-sm font-semibold">
          {visuals.icon}
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-start justify-between gap-3">
            <p className="text-sm font-semibold">{visuals.title}</p>
            {config.dismissible ? (
              <button
                type="button"
                onClick={() => toast.dismiss(toastInstance.id)}
                className="rounded-md border border-current/20 bg-white/70 px-2 py-1 text-[11px] font-medium"
                aria-label="Dismiss toast"
              >
                Dismiss
              </button>
            ) : null}
          </div>
          <p className="mt-1 text-sm leading-5">{visuals.message}</p>
        </div>
      </div>
    ),
    {
      id: LANDING_PAGE_DEMO_TOAST_ID,
      duration: config.dismissible ? Infinity : config.durationMs,
      position: "top-center",
    }
  );
}

/**
 * Landing-page feedback surface that owns demo toast query params.
 */
export default function UIFeedback() {
  const searchParams = useSearchParams();
  const lastDemoToastRef = useRef<string | null>(null);
  const { kind: demoToastKind, config: demoToastConfig, signature: demoToastSignature } =
    resolveDemoToastConfig(searchParams);

  useEffect(() => {
    if (demoToastSignature === lastDemoToastRef.current) return;

    lastDemoToastRef.current = demoToastSignature;

    if (!demoToastKind) {
      toast.dismiss(LANDING_PAGE_DEMO_TOAST_ID);
      return;
    }

    showDemoToast(demoToastKind, demoToastConfig);
  }, [demoToastConfig, demoToastKind, demoToastSignature]);

  return null;
}
