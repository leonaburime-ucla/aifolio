# Behavior Spec: Agentic Research

Spec ID: `agentic-research.behavior`
Version: `1.1.0`
Status: `draft`
Last updated: `2026-02-23`

## Scenario Matrix

1. Scenario: Initial manifest load
- Input: `selectedDatasetId=null`, manifest response includes datasets `[A,B]`
- Steps: invoke manifest load
- Expected output: `datasetManifest=[A,B]`, `selectedDatasetId='A'`

2. Scenario: Dataset selection change
- Input: selected dataset changes from `A` to `B`
- Steps: invoke `setSelectedDatasetId('B')`, run dataset load effect
- Expected output: stale table/chart numeric state cleared before new rows/columns applied

3. Scenario: Active chart precedence
- Input: `pcaChartSpec=null`, `chartSpecs=[X,Y]`
- Steps: compute orchestrator model
- Expected output: `activeChartSpec=X`

4. Scenario: Tool dataset switch
- Input: tool `ar-set_active_dataset` with valid id `B`
- Steps: handler executes
- Expected output: chart store cleared, `selectedDatasetId='B'`, status `ok`

5. Scenario: Invalid remove chart
- Input: `chart_id='missing'`, current ids `[A,B]`
- Steps: call `ar-remove_chart_spec`
- Expected output: `{status:'error', code:'CHART_NOT_FOUND', available_chart_ids:['A','B']}`
