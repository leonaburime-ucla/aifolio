import type { NextConfig } from "next";
import path from "node:path";

const repoRoot = path.resolve(process.cwd(), "../..");

const nextConfig: NextConfig = {
  outputFileTracingRoot: repoRoot,
  turbopack: {
    root: repoRoot,
  },
  transpilePackages: ["@aifolio/contracts", "@aifolio/frontend-core"],
  typescript: {
    ignoreBuildErrors: true,
  },
};

export default nextConfig;
