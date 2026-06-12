import { describe, expect, it } from "vitest";
import { resolveDefaultDatasetId } from "@aifolio/frontend-core/agentic-research";

describe("REQ-001 default dataset selection", () => {
  it("uses selectedDatasetId, then customer churn, then first dataset id, then null", () => {
    expect(
      resolveDefaultDatasetId({
        selectedDatasetId: "already-selected",
        datasets: [{ id: "customer_churn_telco.csv", label: "Churn" }],
      })
    ).toBe("already-selected");

    expect(
      resolveDefaultDatasetId({
        selectedDatasetId: null,
        datasets: [
          { id: "fraud_detection_phishing_websites.csv", label: "Fraud" },
          { id: "customer_churn_telco.csv", label: "Churn" },
        ],
      })
    ).toBe("customer_churn_telco.csv");

    expect(
      resolveDefaultDatasetId({
        selectedDatasetId: null,
        datasets: [{ id: "a", label: "A" }, { id: "b", label: "B" }],
      })
    ).toBe("a");

    expect(
      resolveDefaultDatasetId({
        selectedDatasetId: null,
        datasets: [],
      })
    ).toBeNull();
  });
});
