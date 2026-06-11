import type { EnsureFrameworkTabArgs } from "@aifolio/contracts/entities/ag-ui";

export type MlToolFlowRuntime = {
  querySelector?: (selector: string) => Element | null;
  delay?: (ms: number) => Promise<void>;
  nextFrame?: () => Promise<void>;
};

function defaultDelay(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

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
  }

  await waitForFrameworkForm();
}
