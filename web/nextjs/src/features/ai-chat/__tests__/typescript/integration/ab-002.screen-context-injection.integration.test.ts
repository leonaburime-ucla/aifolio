import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

const orchestratorPath = path.resolve(
  process.cwd(),
  "src/features/ai-chat/typescript/react/orchestrators/chatOrchestrator.ts"
);

describe("AB-002 screen-level context injection", () => {
  it("keeps feature orchestrator dataset context neutral (null)", () => {
    const source = fs.readFileSync(orchestratorPath, "utf8");
    expect(source.includes("activeDatasetId: null")).toBe(true);
  });

  it("does not hardcode screen route context in ai-chat feature", () => {
    const source = fs.readFileSync(orchestratorPath, "utf8");
    const hasRouteLiteral = (route: string): boolean =>
      source.includes(`\"${route}\"`) ||
      source.includes(`'${route}'`) ||
      source.includes(`\`${route}\``);

    expect(hasRouteLiteral("/agentic-research")).toBe(false);
    expect(hasRouteLiteral("/chat")).toBe(false);
    expect(hasRouteLiteral("/landing")).toBe(false);
  });
});
