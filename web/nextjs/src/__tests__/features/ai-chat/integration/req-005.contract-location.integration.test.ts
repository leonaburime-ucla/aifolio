import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

const FEATURE_ROOT = path.resolve(
  process.cwd(),
  "src/features/ai-chat"
);

const CONTRACTS_PKG_ROOT = path.resolve(
  process.cwd(),
  "../packages/contracts/src/entities"
);

function walkFiles(dir: string): string[] {
  if (!fs.existsSync(dir)) return [];
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

describe("REQ-005 contract location — shared packages", () => {
  it("shared contracts package contains chat UI, chat API, and chart entities", () => {
    expect(fs.existsSync(path.join(CONTRACTS_PKG_ROOT, "chat/index.ts"))).toBe(true);
    expect(fs.existsSync(path.join(CONTRACTS_PKG_ROOT, "chat/api/types.ts"))).toBe(true);
    expect(fs.existsSync(path.join(CONTRACTS_PKG_ROOT, "chart/index.ts"))).toBe(true);
    const chatContracts = fs.readFileSync(
      path.join(CONTRACTS_PKG_ROOT, "chat/model/types.ts"),
      "utf8"
    );
    expect(chatContracts).toContain("export type ChatIntegration");
    expect(chatContracts).toContain("export type ChatUiState");
  });

  it("does not keep dead local contract shim files", () => {
    expect(fs.existsSync(path.join(FEATURE_ROOT, "__types__"))).toBe(false);
    expect(fs.existsSync(path.join(FEATURE_ROOT, "__types__/api.types.ts"))).toBe(false);
    expect(fs.existsSync(path.join(FEATURE_ROOT, "__types__/logic"))).toBe(false);
    expect(fs.existsSync(path.join(FEATURE_ROOT, "logic"))).toBe(false);
  });

  it("source files do not import from legacy shim paths", () => {
    const allFiles = walkFiles(FEATURE_ROOT);
    const sourceFiles = allFiles.filter((filePath) =>
      /\.(ts|tsx)$/.test(filePath)
    );

    for (const filePath of sourceFiles) {
      const source = fs.readFileSync(filePath, "utf8");

      const aliasedImports = Array.from(
        source.matchAll(/from\s+["'](@\/[^"']+)["']/g),
        (match) => match[1]
      );

      const legacyImports = aliasedImports.filter(
        (importPath) =>
          importPath.includes("@/features/ai-chat/types/") ||
          importPath.includes("@/features/ai-chat/logic/") ||
          importPath.includes("@/features/ai-chat/__types__") ||
          importPath.includes("@/features/ai-chat/__types__/api.types") ||
          importPath.includes("@/features/ai-chat/__types__/logic/") ||
          importPath === "@/features/charts/contracts/chart.types"
      );

      expect(
        legacyImports,
        `Legacy shim imports found in ${filePath}: ${legacyImports.join(", ")}`
      ).toHaveLength(0);
    }
  });
});
