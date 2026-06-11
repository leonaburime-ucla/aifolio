import { existsSync } from "node:fs";
import path from "node:path";

const runtimePath = path.join(
  process.cwd(),
  "node_modules",
  "next",
  "dist",
  "compiled",
  "next-server",
  "server.runtime.prod.js"
);

if (!existsSync(runtimePath)) {
  console.error(`Missing Next runtime required by Vercel: ${runtimePath}`);
  process.exit(1);
}

console.log(`Verified Next runtime for Vercel: ${runtimePath}`);
