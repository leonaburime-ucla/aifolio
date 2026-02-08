import Papa from "papaparse";
import * as XLSX from "xlsx";

const MAX_PCA_ROWS = 1500;
const NUMERIC_THRESHOLD = 0.9;

/**
 * Row shape returned after parsing CSV/XLS/XLSX files.
 */
export type ParsedRow = Record<string, string | number | null>;

/**
 * Heuristically detect a CSV delimiter from the first line.
 * @param sample - CSV text sample to inspect.
 * @returns The detected delimiter or undefined when unknown.
 */
function detectDelimiter(sample: string): string | undefined {
  const firstLine = sample.split(/\r?\n/)[0] ?? "";
  const counts = new Map<string, number>([
    [",", 0],
    [";", 0],
    ["\t", 0],
  ]);
  for (const delimiter of counts.keys()) {
    counts.set(delimiter, firstLine.split(delimiter).length - 1);
  }
  let best: string | undefined;
  let max = 0;
  for (const [delimiter, count] of counts.entries()) {
    if (count > max) {
      max = count;
      best = delimiter;
    }
  }
  return max > 0 ? best : undefined;
}

/**
 * Parse CSV text into row objects with inferred numeric types.
 * @param text - Raw CSV text.
 * @returns Parsed row objects.
 */
export function parseCsv(text: string): ParsedRow[] {
  const delimiter = detectDelimiter(text);
  const parsed = Papa.parse<ParsedRow>(text, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
    delimiter,
  });
  return parsed.data ?? [];
}

/**
 * Parse XLS/XLSX array buffers into row objects (first sheet only).
 * @param buffer - XLS/XLSX file contents as array buffer.
 * @returns Parsed row objects.
 */
export async function parseXls(buffer: ArrayBuffer): Promise<ParsedRow[]> {
  const workbook = XLSX.read(buffer, { type: "array" });
  const firstSheet = workbook.SheetNames[0];
  if (!firstSheet) return [];
  const sheet = workbook.Sheets[firstSheet];
  return XLSX.utils.sheet_to_json<ParsedRow>(sheet, { defval: null });
}

/**
 * Normalize row keys by trimming whitespace and stripping BOM characters.
 * @param rows - Parsed row objects.
 * @returns Rows with normalized keys.
 */
export function normalizeRowKeys(rows: ParsedRow[]): ParsedRow[] {
  return rows.map((row) => {
    const next: ParsedRow = {};
    Object.entries(row).forEach(([key, value]) => {
      const cleanKey = key.replace(/^\uFEFF/, "").trim();
      if (!cleanKey) return;
      next[cleanKey] = value;
    });
    return next;
  });
}

/**
 * Extract column keys from a sample of rows (default: first 50).
 * @param rows - Parsed row objects.
 * @param sampleSize - Number of rows to sample for column discovery.
 * @returns Ordered list of column keys.
 */
export function getColumnsFromRows(rows: ParsedRow[], sampleSize = 50): string[] {
  const keys = new Set<string>();
  rows.slice(0, sampleSize).forEach((row) => {
    Object.keys(row).forEach((key) => keys.add(key));
  });
  return Array.from(keys);
}

/**
 * Extract numeric-only matrix and feature names from parsed rows.
 * @param rows - Parsed row objects.
 * @returns Numeric matrix and feature name list.
 */
export function extractNumericMatrix(rows: ParsedRow[]) {
  if (rows.length === 0) return { matrix: [], featureNames: [] };
  const keys = Object.keys(rows[0]);
  const numericKeys = keys.filter((key) => {
    let numericCount = 0;
    let totalCount = 0;
    for (const row of rows) {
      if (!(key in row)) continue;
      totalCount += 1;
      const value = row[key];
      if (typeof value === "number" && Number.isFinite(value)) {
        numericCount += 1;
      }
    }
    return totalCount > 0 && numericCount / totalCount >= NUMERIC_THRESHOLD;
  });

  const matrix: number[][] = [];
  for (const row of rows) {
    const values: number[] = [];
    let valid = true;
    for (const key of numericKeys) {
      const value = row[key];
      if (typeof value !== "number" || !Number.isFinite(value)) {
        valid = false;
        break;
      }
      values.push(value);
    }
    if (!valid) continue;
    matrix.push(values);
    if (matrix.length >= MAX_PCA_ROWS) break;
  }

  return { matrix, featureNames: numericKeys };
}

/**
 * Return the lowercase file extension for a path.
 * @param path - File path or URL.
 * @returns Lowercase file extension (no dot).
 */
export function getFileExtension(path: string): string {
  return path.split(".").pop()?.toLowerCase() ?? "";
}
