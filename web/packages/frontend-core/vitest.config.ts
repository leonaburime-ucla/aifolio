import { defineConfig } from "vitest/config";
import path from "path";

export default defineConfig({
  resolve: {
    alias: {
      "@aifolio/contracts": path.resolve(__dirname, "../contracts/src"),
      "@aifolio/contracts/entities/chart": path.resolve(__dirname, "../contracts/src/entities/chart/index.ts"),
      "@aifolio/contracts/entities/chat": path.resolve(__dirname, "../contracts/src/entities/chat/index.ts"),
      "@aifolio/contracts/entities/chat/api": path.resolve(__dirname, "../contracts/src/entities/chat/api.types.ts"),
    },
  },
  test: {
    include: ["__tests__/**/*.test.ts"],
    coverage: {
      provider: "v8",
      include: ["src/**/*.ts"],
      exclude: ["src/**/index.ts"],
      thresholds: {
        statements: 95,
        branches: 95,
        functions: 95,
        lines: 95,
        perFile: true,
      },
    },
  },
});
