import Papa from "papaparse";
import * as XLSX from "xlsx";
import { describe, expect, it, vi } from "vitest";
import {
  extractNumericMatrix,
  getColumnsFromRows,
  getFileExtension,
  normalizeRowKeys,
  parseCsv,
  parseXls,
} from "@/features/agentic-research/typescript/utils/datatable.util";

describe("datatable.util", () => {
  it("parseCsv infers delimiters and dynamic numeric typing", () => {
    expect(parseCsv("a;b\n1;2\n3;4")).toEqual([
      { a: 1, b: 2 },
      { a: 3, b: 4 },
    ]);
    expect(parseCsv("a\tb\n5\t6")).toEqual([{ a: 5, b: 6 }]);
    expect(parseCsv("a,b\n7,8")).toEqual([{ a: 7, b: 8 }]);
    expect(parseCsv("a\n9\n10")).toEqual([{ a: 9 }, { a: 10 }]);
  });

  it("parseCsv falls back to empty array when parser data is undefined", () => {
    const parseSpy = vi
      .spyOn(Papa, "parse")
      .mockReturnValueOnce({ data: undefined } as unknown as ReturnType<typeof Papa.parse>);

    expect(parseCsv("a,b\n1,2")).toEqual([]);
    parseSpy.mockRestore();
  });

  it("parseXls reads first worksheet into objects", async () => {
    const sheet = XLSX.utils.json_to_sheet([
      { a: 1, b: "x" },
      { a: 2, b: "y" },
    ]);
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, sheet, "Sheet1");
    const out = XLSX.write(wb, { bookType: "xlsx", type: "array" }) as ArrayBuffer;

    await expect(parseXls(out)).resolves.toEqual([
      { a: 1, b: "x" },
      { a: 2, b: "y" },
    ]);
  });

  it("normalizes row keys and discovers columns", () => {
    const rows = normalizeRowKeys([
      { "\uFEFF a ": 1, " ": "drop", b: 2 },
      { a: 3, c: 4 },
    ]);

    expect(rows).toEqual([
      { a: 1, b: 2 },
      { a: 3, c: 4 },
    ]);
    expect(getColumnsFromRows(rows)).toEqual(["a", "b", "c"]);
    expect(getColumnsFromRows(rows, 1)).toEqual(["a", "b"]);
  });

  it("extractNumericMatrix enforces numeric threshold and filters invalid rows", () => {
    const rows = [
      { a: 1, b: 2, c: "x", d: 4 },
      { a: 2, b: 3, c: "y", d: 5 },
      { a: 3, b: null, c: "z", d: 6 },
      { a: 4, b: 5, c: "w", d: 7 },
      { a: 5, b: 6, c: "q", d: Infinity },
      { a: 6, b: 7, c: "p", d: 8 },
      { a: 7, b: 8, c: "o", d: 9 },
      { a: 8, b: 9, c: "n", d: 10 },
      { a: 9, b: 10, c: "m", d: 11 },
      { a: 10, b: 11, c: "l", d: 12 },
    ];

    const { matrix, featureNames } = extractNumericMatrix(rows as Array<Record<string, string | number | null>>);
    expect(featureNames).toEqual(["a", "b", "d"]);
    expect(matrix).toEqual([
      [1, 2, 4],
      [2, 3, 5],
      [4, 5, 7],
      [6, 7, 8],
      [7, 8, 9],
      [8, 9, 10],
      [9, 10, 11],
      [10, 11, 12],
    ]);

    expect(extractNumericMatrix([])).toEqual({ matrix: [], featureNames: [] });
  });

  it("extractNumericMatrix handles sparse keys and enforces row cap", () => {
    const sparseRows: Array<Record<string, string | number | null>> = [
      { a: 1, b: 2 },
      { a: 2 },
      { a: 3, b: 4 },
      { a: 4, b: 5 },
      { a: 5, b: 6 },
      { a: 6, b: 7 },
      { a: 7, b: 8 },
      { a: 8, b: 9 },
      { a: 9, b: 10 },
      { a: 10, b: 11 },
    ];
    const sparse = extractNumericMatrix(sparseRows);
    expect(sparse.featureNames).toEqual(["a", "b"]);
    expect(sparse.matrix).toEqual([
      [1, 2],
      [3, 4],
      [4, 5],
      [5, 6],
      [6, 7],
      [7, 8],
      [8, 9],
      [9, 10],
      [10, 11],
    ]);

    const cappedRows = Array.from({ length: 1600 }, (_, index) => ({ a: index + 1 }));
    const capped = extractNumericMatrix(cappedRows);
    expect(capped.featureNames).toEqual(["a"]);
    expect(capped.matrix).toHaveLength(1500);
    expect(capped.matrix[0]).toEqual([1]);
    expect(capped.matrix[1499]).toEqual([1500]);
  });

  it("returns lowercase extension or empty string", () => {
    expect(getFileExtension("/tmp/data.CSV")).toBe("csv");
    expect(getFileExtension("noext")).toBe("noext");
  });
});
