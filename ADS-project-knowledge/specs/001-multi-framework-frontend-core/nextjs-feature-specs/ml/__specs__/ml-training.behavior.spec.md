# Behavior Spec: ML Training

Spec ID: `ml-training.behavior`
Version: `1.1.0`
Status: `draft`
Last updated: `2026-02-23`

## Scenario Matrix

1. Scenario: Manifest fallback selection
- Input: `selectedDatasetId=null`, fetched options `[A,B]`
- Steps: run `useMlDatasetOrchestrator` manifest load
- Expected output: selected id becomes `A`

2. Scenario: Dataset cache short-circuit
- Input: selected id `A` already in `datasetCache`
- Steps: trigger dataset load effect
- Expected output: no API fetch for rows; existing cached rows/columns returned

3. Scenario: PyTorch training failure row
- Input: one combination, `trainModel -> {status:'error', error:'boom'}`
- Steps: run `runPytorchTraining`
- Expected output: one prepended row with `result='failed'` and `error='boom'`

4. Scenario: TensorFlow linear baseline training
- Input: one combination, `isLinearBaselineMode=true`
- Steps: run `runTensorflowTraining`
- Expected output: request payload uses `hidden_dim=128`, `num_hidden_layers=2`, `dropout=0.1`

5. Scenario: Distillation error branch
- Input: distill API returns `{status:'error', error:'x'}`
- Steps: run distillation orchestrator
- Expected output: orchestrator returns `{status:'error', error:'x'}`
