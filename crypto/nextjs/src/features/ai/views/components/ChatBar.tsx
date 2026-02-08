"use client";

import { useChatOrchestrator } from "@/features/ai/orchestrators/chatOrchestrator";

const TOOLTIP_TEXT = "Disabled for now";

type ChatBarProps = {
  mode?: "fixed" | "embedded";
};

/**
 * Bottom-anchored chat input modeled after OpenAI/Claude style.
 */
export default function ChatBar({ mode = "fixed" }: ChatBarProps) {
  const {
    value,
    showTooltip,
    setShowTooltip,
    setValue,
    submit,
    handleHistory,
    resetHistoryCursor,
    isSending,
  } = useChatOrchestrator();

  return (
    <div
      className={
        mode === "fixed"
          ? "fixed inset-x-0 bottom-0 z-50 px-4 pb-6"
          : "w-full h-full"
      }
    >
      <div
        className={
          mode === "fixed"
            ? "mx-auto flex w-full max-w-3xl items-center gap-3 rounded-2xl border border-zinc-200 bg-white/90 px-4 py-3 shadow-lg backdrop-blur"
            : "flex w-full h-full items-center gap-3 bg-white px-4 py-3"
        }
      >
        <div className="relative">
          <button
            type="button"
            aria-disabled="true"
            onClick={() => {
              setShowTooltip(true);
              window.setTimeout(() => setShowTooltip(false), 1500);
            }}
            onMouseEnter={() => setShowTooltip(true)}
            onMouseLeave={() => setShowTooltip(false)}
            className="flex h-10 w-10 items-center justify-center rounded-full border border-zinc-200 bg-zinc-100 text-zinc-400"
          >
            <span className="text-lg">+</span>
          </button>
          <div
            className={`pointer-events-none absolute left-1/2 top-0 -translate-x-1/2 -translate-y-full rounded-md bg-zinc-900 px-2 py-1 text-xs text-white transition-opacity ${showTooltip ? "opacity-100" : "opacity-0"
              }`}
          >
            {TOOLTIP_TEXT}
          </div>
        </div>

        <textarea
          value={value}
          onChange={(event) => {
            setValue(event.target.value);
            resetHistoryCursor();
          }}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey && !event.metaKey && !event.ctrlKey) {
              event.preventDefault();
              void submit();
              return;
            }
            if (event.key === "ArrowUp") {
              event.preventDefault();
              handleHistory("up");
              return;
            }
            if (event.key === "ArrowDown") {
              event.preventDefault();
              handleHistory("down");
            }
          }}
          placeholder="Ask anything"
          rows={mode === "embedded" ? 3 : 1}
          className={`w-full resize-none bg-transparent text-base text-zinc-900 outline-none placeholder:text-zinc-400 ${mode === "embedded" ? "min-h-[6rem] max-h-60" : "min-h-[3rem] max-h-36"
            }`}
          aria-label="Chat input"
        />

        <button
          type="button"
          onClick={() => void submit()}
          disabled={isSending}
          className="rounded-full bg-zinc-900 px-4 py-2 text-sm font-medium text-white transition hover:bg-zinc-800 disabled:cursor-not-allowed disabled:bg-zinc-500"
        >
          Send
        </button>
      </div>
    </div>
  );
}
