import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ChartSpec } from "@aifolio/contracts/entities/chart";
import type { MessageInfo } from "@/features/ag-ui-chat/react/orchestrators/copilotChartBridge.orchestrator.types";

let messages: unknown[] = [];
let activeTab: "charts" | "agentic-research" = "charts";
const addGlobalChartSpec = vi.fn();
const addAgenticChartSpec = vi.fn();

vi.mock("@copilotkit/react-core", () => ({
  useCopilotChatInternal: () => ({
    messages,
  }),
}));

vi.mock("@/features/recharts/react/ai/state/adapters/chartActions.adapter", () => ({
  useCopilotChartActionsAdapter: () => ({
    addChartSpec: addGlobalChartSpec,
  }),
}));

vi.mock("@/features/agentic-research/react/state/adapters/chartActions.adapter", () => ({
  useAgenticResearchChartActionsAdapter: () => ({
    addChartSpec: addAgenticChartSpec,
  }),
}));

vi.mock("@/features/ag-ui-chat/react/state/adapters/agUiWorkspaceState.adapter", () => ({
  useAgUiWorkspaceStateAdapter: () => ({
    activeTab,
  }),
}));

import {
  extractMessageInfo,
  processMessageForChartSpec,
  useCopilotChartBridgeOrchestrator,
} from "@/features/ag-ui-chat/react/orchestrators/copilotChartBridge.orchestrator";

const chartSpec: ChartSpec = {
  id: "chart-a",
  title: "Revenue",
  type: "line",
  xKey: "month",
  yKeys: ["revenue"],
  data: [{ month: "Jan", revenue: 10 }],
};

function assistantContent(spec: ChartSpec | ChartSpec[] | null = chartSpec) {
  return JSON.stringify({
    message: "Here is a chart.",
    chartSpec: spec,
  });
}

function messageInfo(overrides: Partial<MessageInfo> = {}): MessageInfo {
  return {
    messageId: "m1",
    messageType: "TextMessage",
    isTextLike: true,
    messageRole: "assistant",
    messageStatus: "complete",
    messageContent: assistantContent(),
    ...overrides,
  };
}

describe("processMessageForChartSpec", () => {
  beforeEach(() => {
    vi.spyOn(console, "log").mockImplementation(() => {});
  });

  it("skips missing/non-text, already processed, and non-assistant messages", () => {
    expect(processMessageForChartSpec(messageInfo({ messageId: "" }), false)).toEqual({
      result: { status: "skipped", reason: "non_text_or_missing_id" },
      chartSpecs: null,
    });
    expect(processMessageForChartSpec(messageInfo(), true)).toEqual({
      result: { status: "skipped", reason: "already_processed" },
      chartSpecs: null,
    });
    expect(processMessageForChartSpec(messageInfo({ messageRole: "user" }), false)).toEqual({
      result: { status: "skipped", reason: "non_assistant" },
      chartSpecs: null,
    });
  });

  it("waits for streaming content and ignores assistant payloads without charts", () => {
    expect(processMessageForChartSpec(messageInfo({ messageContent: "still streaming {" }), false)).toEqual({
      result: { status: "waiting", reason: "waiting_for_parseable_payload" },
      chartSpecs: null,
    });
    expect(
      processMessageForChartSpec(
        messageInfo({ messageContent: JSON.stringify({ message: "plain answer" }) }),
        false
      )
    ).toEqual({
      result: { status: "no_chart_spec" },
      chartSpecs: null,
    });
  });

  it("normalizes single and multiple chart specs from assistant payloads", () => {
    const single = processMessageForChartSpec(messageInfo(), false);
    const multiple = processMessageForChartSpec(
      messageInfo({
        messageContent: assistantContent([
          chartSpec,
          { ...chartSpec, id: "chart-b", type: "bar" },
        ]),
      }),
      false
    );

    expect(single.result).toEqual({
      status: "charts_added",
      count: 1,
      ids: ["chart-a"],
      types: ["line"],
    });
    expect(single.chartSpecs).toEqual([chartSpec]);
    expect(multiple.result).toEqual({
      status: "charts_added",
      count: 2,
      ids: ["chart-a", "chart-b"],
      types: ["line", "bar"],
    });
  });
});

describe("extractMessageInfo", () => {
  it("normalizes string and array message content", () => {
    expect(extractMessageInfo(null)).toBeNull();
    expect(extractMessageInfo({ type: "TextMessage", role: "assistant", content: "missing id" })).toBeNull();
    expect(
      extractMessageInfo({
        id: "m2",
        type: "TextMessage",
        role: "assistant",
        status: "complete",
        content: ["hello", { text: "world" }, { content: "again" }, { value: "ignored" }],
      })
    ).toEqual({
      messageId: "m2",
      messageType: "TextMessage",
      isTextLike: true,
      messageRole: "assistant",
      messageStatus: "complete",
      messageContent: "hello\nworld\nagain",
    });
  });
});

describe("useCopilotChartBridgeOrchestrator", () => {
  beforeEach(() => {
    messages = [];
    activeTab = "charts";
    addGlobalChartSpec.mockReset();
    addAgenticChartSpec.mockReset();
    vi.spyOn(console, "log").mockImplementation(() => {});
  });

  it("routes assistant chart specs into the global chart store once per message", () => {
    messages = [
      { id: "u1", type: "TextMessage", role: "user", content: "make a chart" },
      { id: "a1", type: "TextMessage", role: "assistant", status: "complete", content: assistantContent() },
    ];

    const { result, rerender } = renderHook(() => useCopilotChartBridgeOrchestrator());
    rerender();

    expect(addGlobalChartSpec).toHaveBeenCalledTimes(1);
    expect(addGlobalChartSpec).toHaveBeenCalledWith(chartSpec);
    expect(addAgenticChartSpec).not.toHaveBeenCalled();

    result.current.processMessage(messageInfo({ messageId: "manual" }));

    expect(addGlobalChartSpec).toHaveBeenCalledTimes(2);
  });

  it("routes chart specs into the agentic research chart store when that tab is active", () => {
    activeTab = "agentic-research";
    messages = [
      { id: "a1", type: "TextMessage", role: "assistant", status: "complete", content: assistantContent() },
    ];

    renderHook(() => useCopilotChartBridgeOrchestrator());

    expect(addAgenticChartSpec).toHaveBeenCalledWith(chartSpec);
    expect(addGlobalChartSpec).not.toHaveBeenCalled();
  });
});
