# Behavior Spec: Recharts

Spec ID: `recharts.behavior`
Version: `1.0.0`
Status: `draft`
Last updated: `2026-02-23`

## Scenario Matrix

1. Scenario: Deduplicating add
- Input: chart ids `[A,B]`, add chart `B`
- Steps: call store `addChartSpec(B)`
- Expected output: chart ids `[B,A]`

2. Scenario: Remove existing chart
- Input: chart ids `[A,B]`, remove `A`
- Steps: call `removeChartSpec('A')`
- Expected output: `[B]`

3. Scenario: Orchestrator passthrough
- Input: injected management port returns custom mocked model
- Steps: call `useChartOrchestrator({useChartManagementPort:mock})`
- Expected output: returned model matches mock output exactly

4. Scenario: Unsupported chart type
- Input: spec type `surface`
- Steps: render `ChartRenderer`
- Expected output: unsupported chart panel is rendered

5. Scenario: Currency formatting
- Input: value `1234.5`, `spec.currency='USD'`
- Steps: call `formatValue`
- Expected output: formatted currency string from `Intl.NumberFormat('en-US',{style:'currency',currency:'USD'})`
