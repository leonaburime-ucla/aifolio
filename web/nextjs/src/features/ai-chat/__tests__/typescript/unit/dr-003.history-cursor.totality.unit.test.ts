import { describe, expect, it } from "vitest";
import { resolveHistoryCursor } from "@/features/ai-chat/typescript/logic/chatStore.logic";

describe("DR-003/INV-003 history cursor totality", () => {
  it("never throws and always returns bounded cursor/value", () => {
    const history = ["one", "two", "three"];
    const cursors = [null, -2, -1, 0, 1, 2, 3, 10] as Array<number | null>;
    const directions = ["up", "down"] as const;

    for (const historyCursor of cursors) {
      for (const direction of directions) {
        expect(() =>
          resolveHistoryCursor({
            inputHistory: history,
            historyCursor,
            direction,
          })
        ).not.toThrow();

        const result = resolveHistoryCursor({
          inputHistory: history,
          historyCursor,
          direction,
        });

        expect(typeof result.value).toBe("string");
        if (result.nextCursor !== null) {
          expect(result.nextCursor).toBeGreaterThanOrEqual(0);
          expect(result.nextCursor).toBeLessThan(history.length);
        }
      }
    }
  });
});
