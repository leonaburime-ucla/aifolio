# Learnings

Use for lessons learned and postmortem-style findings.

## Entries

- YYYY-MM-DD: <what failed/worked and why>
- 2026-03-02: [FAILURE] AG-UI PyTorch compound tool execution was unstable across multiple fixes. Symptoms: repeated follow-up assistant messages, partial action execution, stale preflight errors ("target column not selected" while UI showed selected), and inconsistent sweep/planned-run behavior.
- 2026-03-02: [FAILURE] Attempt: direct DOM field mutation for PyTorch form automation. Result: React controlled inputs diverged from state; values appeared changed but reset on later user edits/renders. Resolution direction: state-first form bridge via React setters.
- 2026-03-02: [FAILURE] Attempt: add dedicated sweep/distill/start fast-paths in backend AG-UI stream. Result: introduced repeated tool-message follow-up loops and transport instability; later reverted.
- 2026-03-02: [FAILURE] Attempt: train intent mapped to `train_pytorch_model` (requires explicit `dataset_id` + `target_column`). Result: generic prompts ("train the model") failed or produced incorrect assistant recovery text. Resolution direction: added `start_pytorch_training_runs` (no args, UI-state-driven).
- 2026-03-02: [FAILURE] Attempt: enforce compound intents (`set_pytorch_form_fields` then `start_pytorch_training_runs`) in backend action normalization. Result: initial enforcement missed pronoun intent ("then train it") and re-injected synthetic actions on follow-up turns, causing loops.
- 2026-03-02: [FAILURE] Attempt: add idempotency guard (skip synthetic enforcement when tool messages already exist after latest user message). Result: reduced loop behavior, but compound sequencing still had state timing races.
- 2026-03-02: [FAILURE] Attempt: bridge-level preflight checks in `startTrainingRuns` for dataset/target. Result: stale closure/render snapshot could read old values and reject training despite UI showing target selected.
- 2026-03-02: [FAILURE] Attempt: use latest callback ref (`onTrainClickRef`) + settle waits (`setTimeout(0)`/`requestAnimationFrame`) between set/start actions. Result: improved some races but not fully deterministic; user still observed missing intended field changes before training.
- 2026-03-02: [FAILURE] Attempt: force `run_sweep=false` unless user explicitly requests sweep. Result: prevented accidental sweep inheritance but exposed remaining issue: requested batch-size changes were not consistently applied before training start.
- 2026-03-02: [SPEC-GAP] No formal action-execution contract exists for Copilot multi-step flows (ordered plan, state-commit acknowledgement, per-step result, stop-on-error semantics). Current architecture mixes LLM-emitted actions, backend action rewriting, and frontend tool execution without a single transaction model.
- 2026-03-02: [OPEN-RISK] Current AG-UI/Copilot tool architecture remains susceptible to race conditions and partial execution for compound commands. Recommended next approach: replace heuristic/race-based sequencing with explicit action transactions and acknowledgements.
- 2026-03-02: [ARCHITECTURE-RULE] Features must not depend directly on other features. `features/*` own vertical-slice domain logic/UI/state; cross-feature wiring belongs in `screens/*`; shared contracts belong in neutral shared/core modules.
