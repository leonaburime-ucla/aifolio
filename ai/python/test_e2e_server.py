"""
End-to-end test for the full pipeline: server → coordinator → data_scientist → analyst.

Run:
  1. Start server: python server.py (in another terminal)
  2. Run tests: python test_e2e_server.py
"""

import json
import requests
import sys

BASE_URL = "http://127.0.0.1:8000"

def test_health():
    """Test server is running."""
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        assert r.status_code == 200
        print("  OK: Server is healthy")
        return True
    except requests.exceptions.ConnectionError:
        print("  FAIL: Server not running. Start with: python server.py")
        return False

def test_sample_datasets():
    """Test sample datasets endpoint."""
    r = requests.get(f"{BASE_URL}/sample-data")
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "ok"
    datasets = data.get("datasets", [])
    print(f"  OK: Found {len(datasets)} datasets")
    return datasets

def test_sklearn_tools():
    """Test sklearn tools endpoint."""
    r = requests.get(f"{BASE_URL}/sklearn-tools")
    assert r.status_code == 200
    data = r.json()
    tools = data.get("tools", [])
    print(f"  OK: Found {len(tools)} sklearn tools")
    return tools

def test_chat_research(dataset_id: str, message: str, expected_chart_type: str = None):
    """Test the full chat-research pipeline."""
    payload = {
        "message": message,
        "dataset_id": dataset_id,
        "model": None,  # Use default
    }

    r = requests.post(f"{BASE_URL}/chat-research", json=payload, timeout=60)
    assert r.status_code == 200, f"Status {r.status_code}: {r.text}"

    data = r.json()
    assert data.get("status") == "ok", f"Status not ok: {data}"

    result = data.get("result", {})

    # Check message exists and is not empty
    msg = result.get("message", "")
    assert msg, "No message in response"

    # Check for [Data Scientist] and [Analyst] sections
    has_ds = "[Data Scientist]" in msg
    has_analyst = "[Analyst]" in msg

    # Check chart spec
    chart_spec = result.get("chartSpec")
    has_chart = chart_spec is not None and len(chart_spec) > 0 if isinstance(chart_spec, list) else chart_spec is not None

    # Check findings
    findings = result.get("findings", [])

    return {
        "message_length": len(msg),
        "has_data_scientist": has_ds,
        "has_analyst": has_analyst,
        "has_chart": has_chart,
        "chart_count": len(chart_spec) if isinstance(chart_spec, list) else (1 if chart_spec else 0),
        "findings_count": len(findings),
        "chart_spec": chart_spec,
        "message_preview": msg[:200] + "..." if len(msg) > 200 else msg,
    }

def run_all_tests():
    print("\n" + "=" * 60)
    print("END-TO-END SERVER TESTS")
    print("=" * 60)

    # Test 1: Health check
    print("\n1. Health Check...")
    if not test_health():
        print("\nServer not running. Exiting.")
        sys.exit(1)

    # Test 2: Sample datasets
    print("\n2. Sample Datasets...")
    datasets = test_sample_datasets()

    # Test 3: Sklearn tools
    print("\n3. Sklearn Tools...")
    tools = test_sklearn_tools()

    # Find wine dataset for testing
    wine_dataset = None
    for ds in datasets:
        if "wine" in ds.get("id", "").lower():
            wine_dataset = ds
            break

    if not wine_dataset:
        print("\n  WARN: No wine dataset found, using first available")
        wine_dataset = datasets[0] if datasets else None

    if not wine_dataset:
        print("\n  FAIL: No datasets available")
        sys.exit(1)

    dataset_id = wine_dataset.get("id")
    print(f"\n  Using dataset: {dataset_id}")

    # Test 4-9: Different algorithm requests
    test_cases = [
        ("PCA", "Run PCA on the data", "scatter"),
        ("PLSR", "Run PLSR to predict quality", "scatter"),
        ("Linear Regression", "Run linear regression to predict quality", "bar"),
        ("Random Forest", "Run random forest regression on quality", "bar"),
        ("Ridge Regression", "Use ridge regression to analyze the data", "bar"),
        ("Gradient Boosting", "Apply gradient boosting regression", "bar"),
    ]

    results = []
    for i, (name, message, expected_type) in enumerate(test_cases, start=4):
        print(f"\n{i}. Testing {name}...")
        print(f"   Query: '{message}'")
        try:
            result = test_chat_research(dataset_id, message, expected_type)
            status = "OK" if result["has_chart"] and result["has_analyst"] else "WARN"
            print(f"   {status}: DS={result['has_data_scientist']}, Analyst={result['has_analyst']}, Charts={result['chart_count']}, Findings={result['findings_count']}")

            # Show chart info if available
            if result["chart_spec"]:
                charts = result["chart_spec"] if isinstance(result["chart_spec"], list) else [result["chart_spec"]]
                for chart in charts[:2]:  # Show first 2
                    if chart is None:
                        print("   WARNING: Chart is None - backend bug!")
                        continue
                    if not isinstance(chart, dict):
                        print(f"   WARNING: Chart is not a dict: {type(chart)}")
                        continue
                    title = chart.get("title", "Unknown")
                    desc = (chart.get("description") or "")[:50]
                    print(f"   Chart: {title} - {desc}..." if desc else f"   Chart: {title}")

            # Show findings preview
            if result["findings_count"] > 0:
                print(f"   Findings preview: {len(result.get('findings', []))} items")

            results.append((name, True, result))
        except Exception as e:
            print(f"   FAIL: {type(e).__name__}: {e}")
            results.append((name, False, str(e)))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"Passed: {passed}/{len(results)}")

    for name, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        if ok:
            d = detail
            print(f"  [{status}] {name}: {d['chart_count']} charts, {d['findings_count']} findings")
        else:
            print(f"  [{status}] {name}: {detail}")

    print("=" * 60)
    return passed == len(results)

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
