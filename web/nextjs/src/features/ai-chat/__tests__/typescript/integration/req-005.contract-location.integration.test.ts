import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

const FEATURE_ROOT = path.resolve(
  process.cwd(),
  "src/features/ai-chat"
);

const REQUIRED_CONTRACT_FILES = [
  "__types__/typescript/chat.types.ts",
  "__types__/typescript/chart.types.ts",
  "__types__/typescript/api.types.ts",
  "__types__/typescript/logic/chatComposition.types.ts",
  "__types__/typescript/logic/chatOrchestrator.types.ts",
  "__types__/typescript/logic/chatStore.types.ts",
  "__types__/typescript/logic/chatSubmission.types.ts",
  "__types__/typescript/logic/modelSelection.types.ts",
] as const;

function walkFiles(dir: string): string[] {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  const files: string[] = [];

  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...walkFiles(fullPath));
      continue;
    }

    files.push(fullPath);
  }

  return files;
}

describe("REQ-005 contract location under __types__/typescript", () => {
  it("keeps required chat contracts in __types__/typescript", () => {
    for (const relativePath of REQUIRED_CONTRACT_FILES) {
      const absolute = path.join(FEATURE_ROOT, relativePath);
      expect(fs.existsSync(absolute)).toBe(true);
    }
  });

  it("does not keep legacy top-level types folder", () => {
    expect(fs.existsSync(path.join(FEATURE_ROOT, "types"))).toBe(false);
  });

  it("imports ai-chat type contracts from __types__/typescript paths", () => {
    const allFiles = walkFiles(path.join(FEATURE_ROOT, "typescript"));
    const sourceFiles = allFiles.filter((filePath) =>
      /\.(ts|tsx)$/.test(filePath)
    );

    for (const filePath of sourceFiles) {
      const source = fs.readFileSync(filePath, "utf8");
      const legacyImport =
        source.includes('"@/features/ai-chat/types/') ||
        source.includes("'@/features/ai-chat/types/");
      expect(legacyImport).toBe(false);

      const typeImportFromFeature =
        source.includes('"@/features/ai-chat/__types__/typescript/') ||
        source.includes("'@/features/ai-chat/__types__/typescript/");

      // Contract-consuming files should reference the __types__ location when using ai-chat types.
      if (source.includes("@/features/ai-chat/") && source.includes("type")) {
        expect(typeImportFromFeature || !source.includes(".types")).toBe(true);
      }
    }
  });
});
