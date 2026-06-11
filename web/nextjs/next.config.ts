import type { NextConfig } from "next";
import path from "node:path";

const workspaceRoot = path.resolve(process.cwd(), "..");

const nextConfig: NextConfig = {
  outputFileTracingRoot: workspaceRoot,
  turbopack: {
    root: workspaceRoot,
  },
  transpilePackages: ["@aifolio/contracts", "@aifolio/frontend-core"],
  typescript: {
    ignoreBuildErrors: true,
  },
};

export default nextConfig;
