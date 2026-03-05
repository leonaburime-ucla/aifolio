import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

const LOGIC_ROOT = path.resolve(process.cwd(), "src/features/ai-chat/typescript/logic");

function walkFiles(dir: string): string[] {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const full = path.join(dir, entry.name);
    return entry.isDirectory() ? walkFiles(full) : [full];
  });
}

describe("AB-003 logic remains framework-agnostic and side-effect-light", () => {
  it("does not import react/zustand/next modules in logic layer", () => {
    const sourceFiles = walkFiles(LOGIC_ROOT).filter((filePath) =>
      /\.(ts|tsx)$/.test(filePath)
    );

    for (const filePath of sourceFiles) {
      const source = fs.readFileSync(filePath, "utf8");
      expect(source.includes('from "react"')).toBe(false);
      expect(source.includes('from "zustand"')).toBe(false);
      expect(source.includes('from "next/')).toBe(false);
    }
  });

  it("does not access browser globals in logic layer", () => {
    const sourceFiles = walkFiles(LOGIC_ROOT).filter((filePath) =>
      /\.(ts|tsx)$/.test(filePath)
    );

    for (const filePath of sourceFiles) {
      const source = fs.readFileSync(filePath, "utf8");
      expect(source.includes("window.")).toBe(false);
      expect(source.includes("document.")).toBe(false);
      expect(source.includes("navigator.")).toBe(false);
    }
  });
});
