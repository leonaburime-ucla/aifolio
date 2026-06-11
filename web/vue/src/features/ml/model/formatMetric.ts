export function formatMetric(val: unknown): string {
  if (typeof val === "number") return val.toFixed(4);
  return String(val ?? "-");
}
