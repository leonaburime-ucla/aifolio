import type { NextConfig } from "next";
import path from "node:path";

const nextConfig: NextConfig = {
  turbopack: {
    root: path.resolve(process.cwd(), ".."),
  },
  transpilePackages: ["@aifolio/contracts", "@aifolio/frontend-core"],
  typescript: {
    ignoreBuildErrors: true,
  },
};

export default nextConfig;
