/**
 * Spec: agentic-research.utils.spec.ts
 * Version: 1.1.0
 */
export const AGENTIC_RESEARCH_UTILS_SPEC_VERSION = "1.1.0";

export const agenticResearchUtilsSpec = {
  id: "agentic-research.utils",
  version: AGENTIC_RESEARCH_UTILS_SPEC_VERSION,
  status: "draft",
  lastUpdated: "2026-02-23",
  units: [
    "parseCsv",
    "parseXls",
    "normalizeRowKeys",
    "getColumnsFromRows",
    "extractNumericMatrix",
    "getFileExtension",
  ],
  requirements: [
    "CSV parsing uses header-based records, dynamic typing, and delimiter auto-detection.",
    "XLS/XLSX parsing reads first sheet only and returns row objects with null default values.",
    "Row-key normalization trims whitespace and strips BOM prefixes from column names.",
    "Numeric matrix extraction includes columns only when numeric ratio >= 0.9 and caps output rows at 1500.",
    "Column discovery samples rows deterministically (default sample size 50).",
  ],
} as const;
