import { describe, expect, it } from "vitest";
import { validateEpochValues } from "@/features/ml/typescript/validators";

describe("validators/index", () => {
  it("re-exports sweep validators", () => {
    const result = validateEpochValues({ raw: "10,20" });
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.values).toEqual([10, 20]);
    }
  });
});
