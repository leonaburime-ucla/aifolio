import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

const FEATURE_ROOT = path.resolve(process.cwd(), "src/features/ai-chat/typescript");

function walkFiles(dir: string): string[] {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const full = path.join(dir, entry.name);
    return entry.isDirectory() ? walkFiles(full) : [full];
  });
}

describe("AB-001 no cross-feature domain imports", () => {
  it("does not import state/orchestrator modules from other features", () => {
    const sourceFiles = walkFiles(FEATURE_ROOT).filter((filePath) =>
      /\.(ts|tsx)$/.test(filePath)
    );

    for (const filePath of sourceFiles) {
      const source = fs.readFileSync(filePath, "utf8");
      const importLines = source
        .split("\n")
        .filter((line) => line.includes("from \"@/features/") || line.includes("from '@/features/"));

      for (const line of importLines) {
        const isAiChatImport = line.includes("@/features/ai-chat/");
        if (isAiChatImport) continue;

        const isDomainLayerImport =
          line.includes("/state/") || line.includes("/orchestrators/");

        expect(isDomainLayerImport).toBe(false);
      }
    }
  });
});
