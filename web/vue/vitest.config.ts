import { defineConfig } from "vitest/config";
import { resolve } from "path";

export default defineConfig({
  test: {
    include: ["__tests__/unit/**/*.spec.ts"],
    environment: "jsdom",
  },
  resolve: {
    alias: {
      "~": resolve(__dirname, "src"),
    },
  },
});
