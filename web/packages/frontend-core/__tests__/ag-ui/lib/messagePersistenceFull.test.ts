import { describe, expect, it } from "vitest";
import {
  toPersistableMessages,
  safeSerialize,
  shouldHydratePersistedMessages,
  shouldSkipEmptyPersistableSync,
} from "../../../src/ag-ui/lib/messagePersistence";

describe("toPersistableMessages", () => {
  it("normalizes a simple user message", () => {
    const result = toPersistableMessages([
      { id: "m1", role: "user", content: "hello" },
    ]);
    expect(result).toEqual([
      { id: "m1", type: "TextMessage", role: "user", content: "hello" },
    ]);
  });

  it("normalizes assistant messages", () => {
    const result = toPersistableMessages([
      { id: "m2", role: "Assistant", content: "hi back" },
    ]);
    expect(result).toEqual([
      { id: "m2", type: "TextMessage", role: "assistant", content: "hi back" },
    ]);
  });

  it("defaults malformed roles to assistant and skips primitive entries", () => {
    const result = toPersistableMessages([
      "primitive entry",
      { id: "m-role", role: 42, content: "default role" },
    ] as never);

    expect(result).toEqual([
      { id: "m-role", type: "TextMessage", role: "assistant", content: "default role" },
    ]);
  });

  it("handles array content with text parts", () => {
    const result = toPersistableMessages([
      {
        id: "m3",
        role: "user",
        content: [{ text: "part1" }, { content: "part2" }],
      },
    ]);
    expect(result[0]).toMatchObject({ content: "part1\npart2" });
  });

  it("handles array content with plain strings", () => {
    const result = toPersistableMessages([
      { id: "m4", role: "user", content: ["line1", "line2"] },
    ]);
    expect(result[0]).toMatchObject({ content: "line1\nline2" });
  });

  it("ignores unsupported array content parts and non-string content", () => {
    expect(
      toPersistableMessages([
        { id: "m7", role: "assistant", content: ["line1", null, { value: "ignored" }, { text: "line2" }] },
      ])
    ).toEqual([
      { id: "m7", type: "TextMessage", role: "assistant", content: "line1\nline2" },
    ]);

    expect(
      toPersistableMessages([
        { id: "m8", role: "assistant", content: { text: "ignored" } },
      ])
    ).toEqual([]);
  });

  it("filters out empty content messages", () => {
    const result = toPersistableMessages([
      { id: "m5", role: "user", content: "" },
      { id: "m6", role: "user", content: "   " },
    ]);
    expect(result).toEqual([]);
  });

  it("filters out messages without id", () => {
    const result = toPersistableMessages([
      { role: "user", content: "no id" },
    ]);
    expect(result).toEqual([]);
  });

  it("filters out coagent-state-render messages", () => {
    const result = toPersistableMessages([
      { id: "coagent-state-render-123", role: "assistant", content: "state" },
    ]);
    expect(result).toEqual([]);
  });

  it("deduplicates by id (last wins)", () => {
    const result = toPersistableMessages([
      { id: "m1", role: "user", content: "first" },
      { id: "m1", role: "user", content: "second" },
    ]);
    expect(result).toHaveLength(1);
    expect(result[0]).toMatchObject({ content: "second" });
  });

  it("strips functions and symbols from values", () => {
    const result = toPersistableMessages([
      { id: "m1", role: "user", content: "test", callback: () => {} },
    ]);
    expect(result[0]).toMatchObject({ id: "m1", content: "test" });
  });

  it("handles circular references gracefully", () => {
    const msg: any = { id: "m1", role: "user", content: "ok" };
    msg.self = msg;
    const result = toPersistableMessages([msg]);
    expect(result).toHaveLength(1);
    expect(result[0]).toMatchObject({ id: "m1", content: "ok" });
  });

  it("returns empty array when message serialization throws", () => {
    const result = toPersistableMessages([
      {
        toJSON() {
          throw new Error("cannot serialize");
        },
      },
    ] as never);

    expect(result).toEqual([]);
  });

  it("returns empty array for non-array input after parse", () => {
    const result = toPersistableMessages([]);
    expect(result).toEqual([]);
  });

  it("returns empty array when serialization is empty or parsed payload is not an array", () => {
    expect(toPersistableMessages(undefined as never)).toEqual([]);
    expect(toPersistableMessages(null as never)).toEqual([]);
  });

  it("handles bigint values by converting to string", () => {
    const result = toPersistableMessages([
      { id: "m1", role: "user", content: "num", bigVal: BigInt(42) },
    ]);
    expect(result).toHaveLength(1);
  });
});

describe("safeSerialize", () => {
  it("serializes a simple object", () => {
    expect(safeSerialize({ a: 1 })).toBe('{"a":1}');
  });

  it("returns empty string for circular references", () => {
    const obj: any = {};
    obj.self = obj;
    expect(safeSerialize(obj)).toBe("");
  });
});

describe("shouldHydratePersistedMessages", () => {
  it("returns false when no persisted messages", () => {
    expect(
      shouldHydratePersistedMessages({
        livePersistableCount: 0,
        liveUserMessageCount: 0,
        persistedCount: 0,
      })
    ).toBe(false);
  });

  it("returns true when no live user messages but have persisted", () => {
    expect(
      shouldHydratePersistedMessages({
        livePersistableCount: 0,
        liveUserMessageCount: 0,
        persistedCount: 5,
      })
    ).toBe(true);
  });

  it("returns true when live user messages exist but no persistable", () => {
    expect(
      shouldHydratePersistedMessages({
        livePersistableCount: 0,
        liveUserMessageCount: 2,
        persistedCount: 3,
      })
    ).toBe(true);
  });

  it("returns false when live persistable messages already exist", () => {
    expect(
      shouldHydratePersistedMessages({
        livePersistableCount: 3,
        liveUserMessageCount: 2,
        persistedCount: 5,
      })
    ).toBe(false);
  });
});

describe("shouldSkipEmptyPersistableSync", () => {
  it("returns true when no live persistable but persisted exist", () => {
    expect(
      shouldSkipEmptyPersistableSync({ livePersistableCount: 0, persistedCount: 5 })
    ).toBe(true);
  });

  it("returns false when live persistable messages exist", () => {
    expect(
      shouldSkipEmptyPersistableSync({ livePersistableCount: 3, persistedCount: 5 })
    ).toBe(false);
  });

  it("returns false when neither live nor persisted exist", () => {
    expect(
      shouldSkipEmptyPersistableSync({ livePersistableCount: 0, persistedCount: 0 })
    ).toBe(false);
  });
});
