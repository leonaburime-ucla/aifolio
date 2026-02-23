import { describe, expect, it, vi } from "vitest";
import * as XLSX from "xlsx";
import Papa from "papaparse";
import {
  extractNumericMatrix,
  getColumnsFromRows,
  getFileExtension,
  normalizeRowKeys,
  parseCsv,
  parseXls,
} from "@/features/agentic-research/utils/datatable.util";

describe("datatable.util", () => {
  it("parseCsv parses semicolon-delimited data", () => {
    const rows = parseCsv("a;b\n1;2\n3;4");
    expect(rows).toEqual([
      { a: 1, b: 2 },
      { a: 3, b: 4 },
    ]);
  });

  it("parseCsv handles input with no known delimiter", () => {
    const rows = parseCsv("just one header\njust one value");
    expect(rows).toEqual([{ "just one header": "just one value" }]);
  });

  it("parseCsv falls back to empty array when parser returns nullish data", () => {
    const parseSpy = vi.spyOn(Papa, "parse").mockReturnValue({
      data: undefined,
      errors: [],
      meta: {
        delimiter: ",",
        linebreak: "\n",
        aborted: false,
        truncated: false,
        cursor: 0,
      },
    } as never);

    const rows = parseCsv("a,b\n1,2");
    expect(rows).toEqual([]);

    parseSpy.mockRestore();
  });

  it("normalizeRowKeys trims and strips BOM", () => {
    const rows = normalizeRowKeys([{ "\uFEFF col ": 1, " other ": 2 }]);
    expect(rows).toEqual([{ col: 1, other: 2 }]);
  });

  it("normalizeRowKeys drops keys that become empty after trimming", () => {
    const rows = normalizeRowKeys([{ "   ": 1, valid: 2 }]);
    expect(rows).toEqual([{ valid: 2 }]);
  });

  it("getColumnsFromRows returns sampled keys", () => {
    const columns = getColumnsFromRows([{ a: 1 }, { b: 2 }, { c: 3 }], 2);
    expect(columns).toEqual(["a", "b"]);
  });

  it("extractNumericMatrix keeps mostly numeric columns", () => {
    const { matrix, featureNames } = extractNumericMatrix([
      { a: 1, b: 2, c: "x" },
      { a: 3, b: 4, c: "y" },
      { a: 5, b: 6, c: "z" },
    ]);
    expect(featureNames).toEqual(["a", "b"]);
    expect(matrix).toEqual([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
  });

  it("extractNumericMatrix returns empty outputs for empty input", () => {
    expect(extractNumericMatrix([])).toEqual({ matrix: [], featureNames: [] });
  });

  it("extractNumericMatrix skips rows with invalid numeric cells", () => {
    const { matrix, featureNames } = extractNumericMatrix([
      { a: 1, b: 2 },
      { a: Number.NaN, b: 4 },
      { a: 5, b: 6 },
    ]);
    expect(featureNames).toEqual(["b"]);
    expect(matrix).toEqual([
      [2],
      [4],
      [6],
    ]);
  });

  it("extractNumericMatrix caps output rows at 1500", () => {
    const rows = Array.from({ length: 1605 }, (_, index) => ({ a: index, b: index + 1 }));
    const { matrix } = extractNumericMatrix(rows);
    expect(matrix).toHaveLength(1500);
  });

  it("extractNumericMatrix skips invalid rows while retaining high-ratio numeric keys", () => {
    const rows = [
      ...Array.from({ length: 9 }, (_, index) => ({ a: index + 1, b: index + 2 })),
      { a: Number.NaN, b: 99 },
    ];
    const { featureNames, matrix } = extractNumericMatrix(rows);
    expect(featureNames).toEqual(["a", "b"]);
    expect(matrix).toHaveLength(9);
  });

  it("extractNumericMatrix handles missing keys in later rows", () => {
    const { featureNames, matrix } = extractNumericMatrix([
      { a: 1, b: 2 },
      { a: 3 },
      { a: 5, b: 6 },
    ]);
    expect(featureNames).toEqual(["a", "b"]);
    expect(matrix).toEqual([
      [1, 2],
      [5, 6],
    ]);
  });

  it("getFileExtension returns lowercase extension", () => {
    expect(getFileExtension("/tmp/data.CSV")).toBe("csv");
  });

  it("getFileExtension returns empty string when no extension exists", () => {
    expect(getFileExtension("/tmp/no-extension")).toBe("/tmp/no-extension");
  });

  it("getFileExtension handles trailing dot", () => {
    expect(getFileExtension("file.")).toBe("");
  });

  it("parseXls parses first sheet rows", async () => {
    const worksheet = XLSX.utils.json_to_sheet([{ a: 1, b: 2 }]);
    const workbook = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(workbook, worksheet, "Sheet1");
    const buffer = XLSX.write(workbook, { type: "array", bookType: "xlsx" }) as ArrayBuffer;

    const rows = await parseXls(buffer);
    expect(rows).toEqual([{ a: 1, b: 2 }]);
  });

});
