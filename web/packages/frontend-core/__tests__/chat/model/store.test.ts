import { describe, it, expect } from "vitest";
import {
  createInitialChatStoreCoreState,
  appendMessage,
  appendInputHistory,
  resolveHistoryCursor,
} from "../../../src/chat/index";

describe("createInitialChatStoreCoreState", () => {
  it("returns default state with all fields defined", () => {
    const state = createInitialChatStoreCoreState({});
    expect(state.messages).toEqual([]);
    expect(state.inputHistory).toEqual([]);
    expect(state.historyCursor).toBeNull();
    expect(state.isSending).toBe(false);
    expect(state.modelOptions).toEqual([]);
    expect(state.selectedModelId).toBeNull();
    expect(state.isModelsLoading).toBe(false);
    expect(state.screenFeedback).toBeNull();
  });

  it("has no undefined fields (INV-06)", () => {
    const state = createInitialChatStoreCoreState({});
    for (const value of Object.values(state)) {
      expect(value).not.toBeUndefined();
    }
  });
});

describe("appendMessage", () => {
  it("appends message to empty array", () => {
    const msg = { id: "1", role: "user" as const, content: "hi", createdAt: 1000 };
    const result = appendMessage({ messages: [], message: msg });
    expect(result).toHaveLength(1);
    expect(result[0]).toBe(msg);
  });

  it("appends message to existing array without mutating", () => {
    const existing = [{ id: "1", role: "user" as const, content: "a", createdAt: 1000 }];
    const msg = { id: "2", role: "assistant" as const, content: "b", createdAt: 2000 };
    const result = appendMessage({ messages: existing, message: msg });
    expect(result).toHaveLength(2);
    expect(existing).toHaveLength(1);
  });
});

describe("appendInputHistory", () => {
  it("appends value and resets cursor to null", () => {
    const result = appendInputHistory({
      inputHistory: ["a", "b"],
      value: "c",
    });
    expect(result.inputHistory).toEqual(["a", "b", "c"]);
    expect(result.historyCursor).toBeNull();
  });
});

describe("resolveHistoryCursor", () => {
  it("returns unchanged cursor for empty history", () => {
    const result = resolveHistoryCursor({
      inputHistory: [],
      historyCursor: null,
      direction: "up",
    });
    expect(result.nextCursor).toBeNull();
    expect(result.value).toBe("");
  });

  it("moves to last item on first up from null cursor", () => {
    const result = resolveHistoryCursor({
      inputHistory: ["a", "b", "c"],
      historyCursor: null,
      direction: "up",
    });
    expect(result.nextCursor).toBe(2);
    expect(result.value).toBe("c");
  });

  it("moves up from existing cursor", () => {
    const result = resolveHistoryCursor({
      inputHistory: ["a", "b", "c"],
      historyCursor: 2,
      direction: "up",
    });
    expect(result.nextCursor).toBe(1);
    expect(result.value).toBe("b");
  });

  it("clamps at 0 when moving up from first item", () => {
    const result = resolveHistoryCursor({
      inputHistory: ["a", "b"],
      historyCursor: 0,
      direction: "up",
    });
    expect(result.nextCursor).toBe(0);
    expect(result.value).toBe("a");
  });

  it("returns null cursor on down from null", () => {
    const result = resolveHistoryCursor({
      inputHistory: ["a", "b"],
      historyCursor: null,
      direction: "down",
    });
    expect(result.nextCursor).toBeNull();
    expect(result.value).toBe("");
  });

  it("moves down and resets cursor at end", () => {
    const result = resolveHistoryCursor({
      inputHistory: ["a", "b"],
      historyCursor: 1,
      direction: "down",
    });
    expect(result.nextCursor).toBeNull();
    expect(result.value).toBe("");
  });

  it("moves down within bounds", () => {
    const result = resolveHistoryCursor({
      inputHistory: ["a", "b", "c"],
      historyCursor: 0,
      direction: "down",
    });
    expect(result.nextCursor).toBe(1);
    expect(result.value).toBe("b");
  });
});
