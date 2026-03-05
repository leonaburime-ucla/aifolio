import { describe, expect, it } from "vitest";
import {
  shouldHydratePersistedMessages,
  shouldSkipEmptyPersistableSync,
} from "@/features/ag-ui-chat/typescript/logic/copilotMessagePersistence.logic";

describe("shouldHydratePersistedMessages", () => {
  it("hydrates when live persistable is empty and persisted exists", () => {
    expect(
      shouldHydratePersistedMessages({
        livePersistableCount: 0,
        liveUserMessageCount: 0,
        persistedCount: 2,
      })
    ).toBe(true);
  });

  it("does not hydrate when live already has persistable messages", () => {
    expect(
      shouldHydratePersistedMessages({
        livePersistableCount: 1,
        liveUserMessageCount: 1,
        persistedCount: 2,
      })
    ).toBe(false);
  });

  it("hydrates when persisted exists and live has only assistant/system snapshots", () => {
    expect(
      shouldHydratePersistedMessages({
        livePersistableCount: 1,
        liveUserMessageCount: 0,
        persistedCount: 2,
      })
    ).toBe(true);
  });
});

describe("shouldSkipEmptyPersistableSync", () => {
  it("skips sync for empty live snapshot when persisted has history", () => {
    expect(
      shouldSkipEmptyPersistableSync({ livePersistableCount: 0, persistedCount: 3 })
    ).toBe(true);
  });

  it("does not skip when persisted is empty", () => {
    expect(
      shouldSkipEmptyPersistableSync({ livePersistableCount: 0, persistedCount: 0 })
    ).toBe(false);
  });
});
