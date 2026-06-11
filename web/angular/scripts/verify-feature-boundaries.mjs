import fs from "node:fs";
import path from "node:path";

const appRoot = path.resolve(process.cwd(), "src/app");
const featureRoot = path.join(appRoot, "features");
const blockedLayers = new Set(["api", "model", "orchestrator"]);
const importPattern = /from\s+["']([^"']+)["']|import\s*\(\s*["']([^"']+)["']\s*\)/g;

function walkFiles(dir) {
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const full = path.join(dir, entry.name);
    return entry.isDirectory() ? walkFiles(full) : [full];
  });
}

function featureInfo(filePath) {
  const rel = path.relative(featureRoot, filePath).split(path.sep);
  if (rel.length < 2 || rel[0].startsWith("..")) return null;
  return { feature: rel[0], layer: rel[1] };
}

function resolveImport(fromFile, specifier) {
  if (!specifier.startsWith(".")) return null;
  const base = path.resolve(path.dirname(fromFile), specifier);
  const candidates = [
    base,
    `${base}.ts`,
    `${base}.tsx`,
    `${base}.js`,
    path.join(base, "index.ts"),
  ];
  return candidates.find((candidate) => fs.existsSync(candidate)) ?? base;
}

const sourceFiles = walkFiles(featureRoot).filter((file) => /\.(ts|tsx)$/.test(file));
const violations = [];

for (const file of sourceFiles) {
  const sourceFeature = featureInfo(file);
  if (!sourceFeature) continue;

  const source = fs.readFileSync(file, "utf8");
  for (const match of source.matchAll(importPattern)) {
    const specifier = match[1] ?? match[2];
    const resolved = resolveImport(file, specifier);
    if (!resolved) continue;

    const targetFeature = featureInfo(resolved);
    if (!targetFeature) continue;
    if (targetFeature.feature === sourceFeature.feature) continue;

    if (blockedLayers.has(targetFeature.layer)) {
      violations.push(
        `${path.relative(process.cwd(), file)} imports ${specifier}, crossing into ${targetFeature.feature}/${targetFeature.layer}`
      );
    }
  }
}

if (violations.length) {
  console.error("Feature boundary violations found:");
  for (const violation of violations) console.error(`- ${violation}`);
  process.exit(1);
}

console.log("Feature boundary guard passed.");
