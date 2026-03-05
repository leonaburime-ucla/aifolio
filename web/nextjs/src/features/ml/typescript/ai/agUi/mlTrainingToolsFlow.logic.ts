import type { EnsureFrameworkTabArgs } from "@/features/ml/__types__/typescript/ai/agUi/mlTrainingTooling.types";

/**
 * Runtime dependencies used for DOM polling and timing in ML tool flows.
 */
export type MlToolFlowRuntime = {
  querySelector?: (selector: string) => Element | null;
  delay?: (ms: number) => Promise<void>;
};

function defaultDelay(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

/**
 * Polls for a framework form field in the DOM.
 */
export async function waitForFrameworkFormField(
  selector: string,
  timeoutMs = 1800,
  runtime: MlToolFlowRuntime = {}
): Promise<boolean> {
  const querySelector = runtime.querySelector ?? document.querySelector.bind(document);
  const delay = runtime.delay ?? defaultDelay;
  const startedAt = Date.now();
  while (Date.now() - startedAt < timeoutMs) {
    const target = querySelector(selector);
    if (target) return true;
    await delay(120);
  }
  return false;
}

/**
 * Ensures AG-UI is on the requested ML framework tab before tool execution.
 */
export async function ensureFrameworkTab({
  activeTab,
  setActiveTab,
  pushRoute,
  frameworkTab,
  waitForFrameworkForm,
}: EnsureFrameworkTabArgs): Promise<void> {
  if (activeTab !== frameworkTab) {
    setActiveTab(frameworkTab);
    pushRoute(`/ag-ui?page=${frameworkTab}`);
    await waitForFrameworkForm();
  }
}
