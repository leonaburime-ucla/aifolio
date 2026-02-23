# Feature Spec: ML Training

Spec ID: `ml-training`
Version: `1.1.0`
Status: `draft`
Last updated: `2026-02-23`

## Scope

In scope:
- ML dataset manifest and rows orchestration.
- PyTorch and TensorFlow training + distillation orchestration.
- Validation and sweep combination generation contracts.
- Training run table row shaping and progress updates.
- Hook integration contracts for PyTorch and TensorFlow pages.

Out of scope:
- Backend trainer implementation internals.
- Route-level page layout styling details.

## Requirement Set

- REQ-001: Dataset manifest load sets selected dataset fallback as `state.selectedDatasetId ?? options[0]?.id ?? null`.
- REQ-002: Dataset rows are cached per dataset id and not re-fetched when cache entry exists.
- REQ-003: Training orchestrators append one run row per combination and call `onProgress(i+1,total)` for each combination.
- REQ-004: Distillation orchestrators return `{status:'error',error}` on downstream error; otherwise return `{status:'ok',metrics,modelId,modelPath,distilledRun}`.
- REQ-005: Sweep validators enforce strict numeric bounds and return user-safe validation errors.

## Deterministic Rules

- DR-001: `buildSweepCombinations` performs a full Cartesian product over all sweep dimensions.
- DR-002: `validateEpochValues` bounds are `[1,500]`, `validateBatchSizes` `[1,200]`, `validateHiddenDims` `[8,500]`, `validateNumHiddenLayers` `[1,15]`, `validateDropouts` `[0,0.9]`, `validateTestSizes` `(0,1)`, learning-rate `(0,1]`.
- DR-003: Failed training rows use `result='failed'` and `metric_name/metric_score/train_loss/test_loss` as `n/a`.
- DR-004: TensorFlow linear baseline mode maps hidden-dim/layers/dropout values to fixed defaults (`128`, `2`, `0.1`) in request payloads.

## Acceptance Scenarios

- AC-001 (REQ-001): Given no selected dataset and options `[A,B]`, when manifest load completes, selected dataset becomes `A`.
- AC-002 (REQ-002): Given cached entry exists for dataset `A`, when selected dataset is `A`, dataset rows API is not called.
- AC-003 (REQ-003): Given 3 combinations, training run invokes progress callbacks with `(1,3)`, `(2,3)`, `(3,3)` and appends 3 rows.
- AC-004 (REQ-004): Given distill API returns error, orchestrator returns `{status:'error'}` and no success payload fields.
- AC-005 (REQ-005): Given `epochValues='0,10'`, validator returns `{ok:false,error:'Epoch out of range (1-500): 0'}`.

## Open Clarifications

- None.
