import { describe, it, expect, vi, beforeEach } from "vitest";
import { nextTick } from "vue";
import { useChatSidebar } from "~/features/ai-chat/model";
import type { ChatApi } from "~/features/ai-chat/api";

vi.mock("@aifolio/frontend-core/chat", () => ({
  sendChatMessageDirect: vi.fn().mockImplementation(async (_msg: any, opts: any) => {
    opts?.runtimeDeps?.resolveBaseUrl?.();
    return { message: "Direct response", chartSpec: null };
  }),
  sendChatMessage: vi.fn().mockImplementation(async (_msg: any, opts: any) => {
    opts?.runtimeDeps?.resolveBaseUrl?.();
    return { message: "Research response", chartSpec: null };
  }),
}));

vi.mock("~/features/ai-chat/api", async (importOriginal) => {
  const orig = await importOriginal<typeof import("~/features/ai-chat/api")>();
  return {
    ...orig,
    createChatApi: vi.fn(() => ({
      fetchModels: vi.fn().mockResolvedValue({
        models: [{ id: "default-model", label: "Default" }],
        currentModel: "default-model",
      }),
    })),
  };
});

function createMockApi(overrides: Partial<ChatApi> = {}): ChatApi {
  return {
    fetchModels: vi.fn().mockResolvedValue({
      models: [
        { id: "gemini-flash", label: "Gemini Flash" },
        { id: "gemini-pro", label: "Gemini Pro" },
      ],
      currentModel: "gemini-flash",
    }),
    ...overrides,
  };
}

describe("useChatSidebar", () => {
  let api: ChatApi;
  let chat: ReturnType<typeof useChatSidebar>;

  beforeEach(() => {
    api = createMockApi();
    chat = useChatSidebar({
      baseUrl: "/api/ai",
      mode: "direct",
      getDatasetId: () => null,
      api,
    });
  });

  describe("loadModels()", () => {
    it("populates model options and selects current", async () => {
      await chat.loadModels();

      expect(chat.modelOptions.value).toHaveLength(2);
      expect(chat.selectedModelId.value).toBe("gemini-flash");
    });

    it("handles fetch failure gracefully", async () => {
      const failApi = createMockApi({
        fetchModels: vi.fn().mockRejectedValue(new Error("down")),
      });
      const c = useChatSidebar({
        baseUrl: "/api/ai",
        mode: "direct",
        getDatasetId: () => null,
        api: failApi,
      });
      await c.loadModels();

      expect(c.modelOptions.value).toHaveLength(0);
      expect(c.selectedModelId.value).toBeNull();
    });
  });

  describe("submit()", () => {
    it("no-ops when input is empty", async () => {
      chat.inputValue.value = "   ";
      await chat.submit();

      expect(chat.messages.value).toHaveLength(0);
    });

    it("no-ops when already sending", async () => {
      chat.isSending.value = true;
      chat.inputValue.value = "hello";
      await chat.submit();

      expect(chat.messages.value).toHaveLength(0);
    });

    it("sends message and appends user + assistant messages", async () => {
      chat.inputValue.value = "hello";
      await chat.submit();

      expect(chat.messages.value).toHaveLength(2);
      expect(chat.messages.value[0].role).toBe("user");
      expect(chat.messages.value[0].content).toBe("hello");
      expect(chat.messages.value[1].role).toBe("assistant");
      expect(chat.messages.value[1].content).toBe("Direct response");
    });

    it("uses sendChatMessage in research mode", async () => {
      const { sendChatMessage } = await import("@aifolio/frontend-core/chat");
      const c = useChatSidebar({
        baseUrl: "/api/ai",
        mode: "research",
        getDatasetId: () => "churn.csv",
        api,
      });
      c.inputValue.value = "analyze";
      await c.submit();

      expect(sendChatMessage).toHaveBeenCalled();
      expect(c.messages.value[1].content).toBe("Research response");
    });

    it("clears input and resets history cursor after send", async () => {
      chat.inputValue.value = "hello";
      await chat.submit();

      expect(chat.inputValue.value).toBe("");
      expect(chat.historyCursor.value).toBeNull();
    });

    it("sets screenFeedback on empty response", async () => {
      const { sendChatMessageDirect } = await import("@aifolio/frontend-core/chat");
      vi.mocked(sendChatMessageDirect).mockResolvedValueOnce(null as any);

      chat.inputValue.value = "hello";
      await chat.submit();

      expect(chat.screenFeedback.value?.code).toBe("CHAT_EMPTY_RESPONSE");
    });

    it("sets screenFeedback on request failure", async () => {
      const { sendChatMessageDirect } = await import("@aifolio/frontend-core/chat");
      vi.mocked(sendChatMessageDirect).mockRejectedValueOnce(new Error("Network error"));

      chat.inputValue.value = "hello";
      await chat.submit();

      expect(chat.screenFeedback.value?.code).toBe("CHAT_REQUEST_FAILED");
      expect(chat.screenFeedback.value?.message).toBe("Network error");
    });

    it("sets generic message on non-Error throw", async () => {
      const { sendChatMessageDirect } = await import("@aifolio/frontend-core/chat");
      vi.mocked(sendChatMessageDirect).mockRejectedValueOnce("string error");

      chat.inputValue.value = "hello";
      await chat.submit();

      expect(chat.screenFeedback.value?.message).toBe("Request failed.");
    });

    it("calls onChartSpec when response includes chart specs", async () => {
      const { sendChatMessageDirect } = await import("@aifolio/frontend-core/chat");
      vi.mocked(sendChatMessageDirect).mockResolvedValueOnce({
        message: "Here's a chart",
        chartSpec: { id: "c1", type: "line", title: "Test", data: [{ x: 1, y: 2 }] },
      } as any);

      const chartSpy = vi.fn();
      const c = useChatSidebar({
        baseUrl: "/api/ai",
        mode: "direct",
        getDatasetId: () => null,
        onChartSpec: chartSpy,
        api,
      });
      c.inputValue.value = "chart";
      await c.submit();

      expect(chartSpy).toHaveBeenCalledWith(
        expect.objectContaining({ id: "c1", type: "line" })
      );
    });

    it("handles array of chart specs", async () => {
      const { sendChatMessageDirect } = await import("@aifolio/frontend-core/chat");
      vi.mocked(sendChatMessageDirect).mockResolvedValueOnce({
        message: "Charts",
        chartSpec: [
          { id: "c1", type: "line", data: [{ x: 1 }] },
          { id: "c2", type: "bar", data: [{ x: 2 }] },
        ],
      } as any);

      const chartSpy = vi.fn();
      const c = useChatSidebar({
        baseUrl: "/api/ai",
        mode: "direct",
        getDatasetId: () => null,
        onChartSpec: chartSpy,
        api,
      });
      c.inputValue.value = "charts";
      await c.submit();

      expect(chartSpy).toHaveBeenCalledTimes(2);
    });

    it("skips chart specs without data field", async () => {
      const { sendChatMessageDirect } = await import("@aifolio/frontend-core/chat");
      vi.mocked(sendChatMessageDirect).mockResolvedValueOnce({
        message: "No data chart",
        chartSpec: { id: "c1", type: "line" },
      } as any);

      const chartSpy = vi.fn();
      const c = useChatSidebar({
        baseUrl: "/api/ai",
        mode: "direct",
        getDatasetId: () => null,
        onChartSpec: chartSpy,
        api,
      });
      c.inputValue.value = "test";
      await c.submit();

      expect(chartSpy).not.toHaveBeenCalled();
    });

    it("resets isSending after completion", async () => {
      chat.inputValue.value = "hello";
      await chat.submit();

      expect(chat.isSending.value).toBe(false);
    });

    it("pushes to input history", async () => {
      chat.inputValue.value = "first";
      await chat.submit();
      chat.inputValue.value = "second";
      await chat.submit();

      expect(chat.inputHistory.value).toEqual(["first", "second"]);
    });
  });

  describe("scrollToBottom()", () => {
    it("scrolls messagesEl when messages change", async () => {
      const el = { scrollTop: 0, scrollHeight: 500 } as unknown as HTMLElement;
      chat.messagesEl.value = el;
      chat.messages.value = [
        ...chat.messages.value,
        { id: "test", role: "user", content: "hi", createdAt: 1 },
      ];
      await nextTick();
      await nextTick();
      await nextTick();

      expect(el.scrollTop).toBe(500);
    });

    it("no-ops when messagesEl is null", async () => {
      chat.messagesEl.value = null;
      chat.messages.value = [
        { id: "test2", role: "user", content: "hi", createdAt: 1 },
      ];
      await nextTick();
      await nextTick();
    });
  });

  describe("api fallback", () => {
    it("uses createChatApi when no api option provided", async () => {
      const c = useChatSidebar({
        baseUrl: "/api/ai",
        mode: "direct",
        getDatasetId: () => null,
      });
      await c.loadModels();
      expect(c.modelOptions.value).toHaveLength(1);
      expect(c.modelOptions.value[0].id).toBe("default-model");
    });
  });

  describe("handleHistory()", () => {
    it("no-ops when history is empty", () => {
      chat.handleHistory("up");
      expect(chat.inputValue.value).toBe("");
    });

    it("navigates up through history", async () => {
      chat.inputValue.value = "first";
      await chat.submit();
      chat.inputValue.value = "second";
      await chat.submit();

      chat.handleHistory("up");
      expect(chat.inputValue.value).toBe("second");

      chat.handleHistory("up");
      expect(chat.inputValue.value).toBe("first");
    });

    it("stops at beginning of history", async () => {
      chat.inputValue.value = "only";
      await chat.submit();

      chat.handleHistory("up");
      chat.handleHistory("up");
      expect(chat.inputValue.value).toBe("only");
      expect(chat.historyCursor.value).toBe(0);
    });

    it("navigates down clears input at end", async () => {
      chat.inputValue.value = "msg";
      await chat.submit();

      chat.handleHistory("up");
      expect(chat.inputValue.value).toBe("msg");

      chat.handleHistory("down");
      expect(chat.inputValue.value).toBe("");
      expect(chat.historyCursor.value).toBeNull();
    });

    it("no-ops down when cursor is null", () => {
      chat.inputHistory.value = ["something"];
      chat.handleHistory("down");
      expect(chat.inputValue.value).toBe("");
    });

    it("navigates down through middle of history", async () => {
      chat.inputValue.value = "a";
      await chat.submit();
      chat.inputValue.value = "b";
      await chat.submit();
      chat.inputValue.value = "c";
      await chat.submit();

      chat.handleHistory("up");
      chat.handleHistory("up");
      chat.handleHistory("up");
      expect(chat.inputValue.value).toBe("a");

      chat.handleHistory("down");
      expect(chat.inputValue.value).toBe("b");
    });
  });
});
