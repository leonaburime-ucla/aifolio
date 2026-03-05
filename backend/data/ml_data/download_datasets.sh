#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAX_BYTES=$((20 * 1024 * 1024))

fetch() {
  local use_case="$1"
  local url="$2"
  local out="$3"
  local min_cols="$4"

  echo "Downloading ${out}"
  curl -fL --retry 2 --connect-timeout 30 "$url" -o "$OUT_DIR/$out"

  local bytes
  bytes="$(wc -c < "$OUT_DIR/$out" | tr -d ' ')"
  if [ "$bytes" -gt "$MAX_BYTES" ]; then
    echo "ERROR: ${out} is larger than 20MB ($bytes bytes)." >&2
    rm -f "$OUT_DIR/$out"
    return 1
  fi

  python3 - "$OUT_DIR/$out" "$min_cols" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
min_cols = int(sys.argv[2])

suffix = path.suffix.lower()
cols = 0
if suffix == ".csv":
    with path.open("r", encoding="utf-8", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        reader = csv.reader(f, dialect=dialect)
        header = next(reader, [])
        cols = len(header)
elif suffix in {".xls", ".xlsx"}:
    # We skip strict column validation for excel files in shell gate.
    cols = min_cols

if cols < min_cols:
    raise SystemExit(f"Dataset has too few columns: {cols} < {min_cols}")

print(f"Validated: {path.name} cols={cols}")
PY

  echo "OK: ${use_case} -> ${out} (${bytes} bytes)"
}

# Customer churn (21 columns)
fetch "customer_churn" \
  "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv" \
  "customer_churn_telco.csv" 8 || \
fetch "customer_churn" \
  "https://raw.githubusercontent.com/blastchar/telco-customer-churn/master/WA_Fn-UseC_-Telco-Customer-Churn.csv" \
  "customer_churn_telco.csv" 8

# House prices (Ames has 80+ columns)
fetch "house_prices" \
  "https://cmustatistics.github.io/data-repository/data/ames-housing.csv" \
  "house_prices_ames.csv" 10 || \
fetch "house_prices" \
  "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv" \
  "house_prices_boston.csv" 8

# Loan default (25 columns, 5.3MB)
fetch "loan_default" \
  "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls" \
  "loan_default_credit_card_clients.xls" 8

# Fraud detection (phishing fraud, 30+ columns)
fetch "fraud_detection" \
  "https://raw.githubusercontent.com/sachinshubhams/Website-Phishing/main/csv_result-Training%20Dataset.csv" \
  "fraud_detection_phishing_websites.csv" 12

# Sales forecasting (10 columns)
fetch "sales_forecasting" \
  "https://huggingface.co/datasets/Ammok/walmart_sales_prediction/resolve/main/Walmart.csv" \
  "sales_forecasting_walmart.csv" 8

cat > "$OUT_DIR/sources.json" <<'JSON'
{
  "customer_churn_telco.csv": [
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/blastchar/telco-customer-churn/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"
  ],
  "house_prices_ames.csv": [
    "https://cmustatistics.github.io/data-repository/data/ames-housing.csv"
  ],
  "house_prices_boston.csv": [
    "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
  ],
  "loan_default_credit_card_clients.xls": [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
  ],
  "fraud_detection_phishing_websites.csv": [
    "https://raw.githubusercontent.com/sachinshubhams/Website-Phishing/main/csv_result-Training%20Dataset.csv"
  ],
  "sales_forecasting_walmart.csv": [
    "https://huggingface.co/datasets/Ammok/walmart_sales_prediction/resolve/main/Walmart.csv"
  ]
}
JSON

echo "Done. Files in $OUT_DIR"
ls -lh "$OUT_DIR"
