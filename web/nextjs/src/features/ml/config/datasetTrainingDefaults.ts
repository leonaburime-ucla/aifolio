export type DatasetTrainingDefaults = {
  targetColumn: string;
  excludeColumns: string[];
  dateColumns: string[];
  task: "classification" | "regression" | "auto";
  epochs: number;
};

const DEFAULTS_BY_DATASET_ID: Record<string, DatasetTrainingDefaults> = {
  "customer_churn_telco.csv": {
    targetColumn: "Churn",
    excludeColumns: ["customerID"],
    dateColumns: [],
    task: "classification",
    epochs: 40,
  },
  "house_prices_ames.csv": {
    targetColumn: "SalePrice",
    excludeColumns: ["Order", "PID"],
    dateColumns: [],
    task: "regression",
    epochs: 80,
  },
  "fraud_detection_phishing_websites.csv": {
    targetColumn: "Result",
    excludeColumns: [],
    dateColumns: [],
    task: "classification",
    epochs: 60,
  },
  "sales_forecasting_walmart.csv": {
    targetColumn: "Weekly_Sales",
    excludeColumns: [],
    dateColumns: ["Date"],
    task: "regression",
    epochs: 80,
  },
  "loan_default_credit_card_clients.xls": {
    targetColumn: "default payment next month",
    excludeColumns: ["ID"],
    dateColumns: [],
    task: "classification",
    epochs: 80,
  },
};

export function getTrainingDefaults(datasetId: string | null): DatasetTrainingDefaults {
  if (datasetId && DEFAULTS_BY_DATASET_ID[datasetId]) {
    return DEFAULTS_BY_DATASET_ID[datasetId];
  }
  return {
    targetColumn: "",
    excludeColumns: [],
    dateColumns: [],
    task: "auto",
    epochs: 60,
  };
}
