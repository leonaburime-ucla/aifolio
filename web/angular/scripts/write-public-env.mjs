import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = dirname(fileURLToPath(import.meta.url));
const outputPath = join(scriptDir, '../src/app/core/config/public-env.generated.ts');
const apiBaseUrl = process.env.ANGULAR_PUBLIC_AI_API_URL?.trim() || '/api/ai';

mkdirSync(dirname(outputPath), { recursive: true });
writeFileSync(
  outputPath,
  `export const ANGULAR_PUBLIC_AI_API_URL = ${JSON.stringify(apiBaseUrl)};\n`
);
