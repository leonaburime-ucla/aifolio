"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { AG_UI_PREFERRED_MODEL_ID } from "@/features/ag-ui-chat/typescript/config/agUiModel.config";
import { useAgUiModelStore } from "@/features/ag-ui-chat/typescript/react/state/zustand/agUiModelStore";

/**
 * Primary site navigation.
 */
export default function Navbar() {
  const pathname = usePathname();
  const setSelectedModelId = useAgUiModelStore((state) => state.setSelectedModelId);
  const isHome = pathname === "/";
  const isAgenticResearch = pathname === "/agentic-research";
  const isMachineLearning = pathname.startsWith("/ml");
  const isAgUi = pathname === "/ag-ui";

  function getNavItemClassName(isActive: boolean): string {
    return `rounded-md px-2 py-1 transition ${
      isActive
        ? "bg-zinc-300 text-zinc-950"
        : "text-zinc-900 hover:bg-zinc-100"
    }`;
  }

  return (
    <nav className="sticky top-0 z-40 border-b border-zinc-200 bg-white">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
        <div className="flex items-center gap-6 text-sm text-zinc-900">
          <Link
            href="/"
            className={getNavItemClassName(isHome)}
          >
            AI Chat
          </Link>

          <Link
            href="/agentic-research"
            className={getNavItemClassName(isAgenticResearch)}
          >
            Agentic Research
          </Link>
          <div className="group relative">
            <button
              type="button"
              className={`flex items-center gap-1 rounded-md px-2 py-1 transition ${
                isMachineLearning
                  ? "bg-zinc-300 text-zinc-950"
                  : "text-zinc-900 hover:bg-zinc-100"
              }`}
            >
              Machine Learning
              <span className="text-xs">▾</span>
            </button>
            <div className="invisible absolute left-0 top-full pt-2 opacity-0 transition group-hover:visible group-hover:opacity-100 group-focus-within:visible group-focus-within:opacity-100">
              <div className="w-48 rounded-lg border border-zinc-200 bg-white p-2 text-sm text-zinc-700 shadow-lg">
                <Link
                  href="/ml/pytorch"
                  className="block rounded-md px-3 py-2 transition hover:bg-zinc-100"
                >
                  PyTorch
                </Link>
                <Link
                  href="/ml/tensorflow"
                  className="block rounded-md px-3 py-2 transition hover:bg-zinc-100"
                >
                  TensorFlow
                </Link>
              </div>
            </div>
          </div>
          <Link
            href="/ag-ui"
            className={getNavItemClassName(isAgUi)}
            onClick={() => setSelectedModelId(AG_UI_PREFERRED_MODEL_ID)}
          >
            Agentic UI
          </Link>
        </div>

        <Link href="/" className="text-sm font-semibold text-zinc-900">
          AIfolio
        </Link>
      </div>
    </nav>
  );
}
