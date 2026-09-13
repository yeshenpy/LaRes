# Debug and Integration Specification

**Project:** LARES symbolic-policy evolution for MetaWorld push  
**Status:** Implementation required; Phase 7 is blocked  
**Purpose:** Repair the gaps between the intended research protocol in `spec.md` and the current implementation before running another large experiment.

## 1. What to do next

Implement and verify the work in this order:

1. Fix end-to-end candidate lineage and LLM-call provenance.
2. Put the objective-aware fitter into the live evolutionary path and make checkpoint reports internally consistent.
3. Strengthen experiment records, resource accounting, archive updates, reconstruction, and gate intervention.
4. Run a two-generation smoke test and require every acceptance check in this document to pass.
5. Run one fresh development-confirmation experiment for the fitting/checkpoint decision.
6. Only then implement and run the correction-LLM repair ablation.

Do **not** start Phase 7, access the final test set, run Qwen/E10, or launch another costly evolutionary study before steps 1–4 pass.

## 2. Ground truth and status corrections

### 2.1 Environment naming

`push-v3` is not a typo. The project-facing task ID is `push-v2`, while MetaWorld 3.0 registers the same environment as `push-v3`. `metaworld_env` intentionally rewrites `-v2` to `-v3` before consulting `ALL_V3_ENVIRONMENTS`.

Every new manifest and run record must store:

- `canonical_env_id: push-v2`
- `resolved_registry_env_id: push-v3`
- the compatibility mapping rule or version

Tests should use `push-v2` at the public API and assert that the resolved registry ID is `push-v3`.

### 2.2 Research phase status

Update the implementation status as follows:

| Phase | Correct status |
|---|---|
| Phase 4 — feedback-guided evolution | Reopen. Feedback text exists, but candidate-level lineage and controlled edit semantics are not integrated. |
| Phase 5 — objective/fitting machinery | Experimental machinery exists; live evolutionary integration is incomplete. |
| Phase 6 — repair | The existing ablation is a pilot. It does not cleanly test the intended LLM repair arms or satisfy final AC6. |
| Phase 7 — final evaluation | Blocked until this specification passes. |

## 3. Verified defects to fix

| ID | Severity | Defect | Required outcome |
|---|---:|---|---|
| D1 | Critical | Generation 1 records have empty `parent_ids`, `idea_id`, and `proposal_id`. | Every new candidate, including failures, has explicit lineage and idea/proposal references. |
| D2 | Critical | Live generation returns parallel lists and discards structured lineage and LLM usage. | Return typed candidate objects with stable IDs and LLM-call references. |
| D3 | Critical | The main fitter still uses legacy `MSE + 0.01 * std.mean()` and trains scale parameters. | Live evolution uses the named deterministic objective and fixed, non-trainable scale. |
| D4 | Critical | A selected checkpoint can be paired with diagnostics and parameters from a different checkpoint. | Selected state, step, metrics, diagnostics, and report always refer to the same checkpoint. |
| D5 | Critical | Serialized records omit the states and evidence needed to reconstruct a result. | Evaluated candidates are reconstructable from record-referenced, hashed artifacts. |
| D6 | High | `environment_steps` currently contains BC gradient steps and omits real simulator work. | Gradient work, simulator steps, episodes, LLM usage, and wall time are recorded separately. |
| D7 | High | Archive entries are created from screen results and not updated after expanded evaluation. | Archive exposes screen and confirmed metrics and uses confirmed metrics when available. |
| D8 | High | Generated policies can record a forced gate without using its returned value in the action. | Every advertised forceable gate demonstrably changes the action when overridden. |
| D9 | High | Missing ideas are silently padded, allowing malformed ideation output to pass. | Invalid or incomplete structured output fails explicitly and is retryable and auditable. |
| D10 | High | The current repair ablation mixes unequal fitting/search budgets and does not exercise correction-LLM repair. | Repair arms share preprocessing and have matched, explicit budgets. |
| D11 | Medium | Summary tooling can claim full traceability while lineage, state, prompt, or resource fields are absent. | Traceability is computed from all required evidence fields and reconstruction status. |
| D12 | Medium | Candidate/failure IDs and repeated attempts are not reliably one-to-one. | Candidate slot and attempt IDs are allocated before any LLM or validation call. |

## 4. Required data contracts

Put shared schemas in `lares/search/schemas.py` or a new dependency-light module such as `lares/search/generation.py`. Do not create a circular import between policy generation and the training pipeline.

### 4.1 Feedback bundle

```python
@dataclass(frozen=True)
class FeedbackBundle:
    feedback_id: str
    source_generation: int
    manifest_id: str
    context_parent_ids: list[str]
    text_path: str
    text_sha256: str
```

`context_parent_ids` means every policy whose evidence was shown to the LLM. It is not a claim that every listed policy was semantically used in every proposal.

### 4.2 Generated candidate

```python
@dataclass
class GeneratedCandidate:
    candidate_id: str
    generation: int
    slot: int
    attempt: int
    policy: object | None
    code: str | None
    response: str | None
    idea: PolicyIdea | None
    proposal: MutationProposal | None
    feedback_id: str | None
    context_parent_ids: list[str]
    llm_call_ids: list[str]
    validation: dict
```

Rules:

- Allocate `candidate_id`, `slot`, and `attempt` before calling the LLM.
- A failed generation or validation attempt keeps the same metadata and is written as a failed record.
- Generation 0 requires a valid `PolicyIdea` and has no semantic parent.
- Generation 1 and later require a valid `MutationProposal` plus the resulting `PolicyIdea`.
- A repair or simplification proposal has exactly one semantic parent.
- A crossover proposal has at least two semantic parents.
- Explicit exploration may have no semantic parent, but must say why it is not using a parent.
- Every semantic `parent_id` must be a member of the feedback bundle's `context_parent_ids`.
- `idea_id`, `proposal_id`, and candidate IDs are assigned by code, not trusted from model output.

### 4.3 LLM-call ledger

Write one run-level ledger entry per actual LLM call:

```yaml
call_id: call_...
label: ideation | generation | correction | repair
model: ...
temperature: ...
decoding: {...}
prompt_path: ...
prompt_sha256: ...
response_path: ...
response_sha256: ...
prompt_tokens: 0
completion_tokens: 0
total_tokens: 0
wall_time_seconds: 0.0
```

Candidate records reference `llm_call_ids`. A shared ideation call is counted once in run totals, not once per candidate.

### 4.4 Fitting and state artifacts

Every fitted candidate must retain:

- `initial_state_path` and hash
- `final_state_path` and hash
- `selected_state_path` and hash
- `selected_step`
- rollout-probe result for step 0 and every configured checkpoint
- final-fit diagnostics computed on `final_state`
- selected-fit diagnostics recomputed after loading `selected_state`
- objective name and full objective configuration
- fixed action-scale value and action mode

The saved record may reference state files rather than embedding tensors, but every reference must be relative to the run root and content-hashed.

### 4.5 Resource ledger

Record these counters separately:

- `bc_gradient_steps`
- `bc_objective_evaluations`
- `probe_episodes` and `probe_environment_steps`
- `screen_episodes` and `screen_environment_steps`
- `expanded_episodes` and `expanded_environment_steps`
- `repair_episodes` and `repair_environment_steps`
- `expert_assisted_episodes` and `expert_assisted_environment_steps`
- `llm_calls`, prompt/completion/total tokens
- wall time by generation, fitting, evaluation, and repair stage

Never store optimizer steps in a field named `environment_steps`.

## 5. Implementation stages

### Stage A — generation and lineage

Primary files:

- `lares/core/policy_generation.py`
- `lares/core/training_pipeline.py`
- `lares/search/schemas.py`
- `lares/search/feedback.py`
- generation, ideation, correction, and feedback prompts

Changes:

1. Replace the live three-parallel-list return contract with `list[GeneratedCandidate]`.
2. Make the ideation prompt return a strict structured schema that can instantiate `PolicyIdea` objects.
3. Make the generation-1+ prompt return a strict `MutationProposal` and resulting idea, including edit mode, parents, measurable prediction, and protected behavior.
4. Reject missing or malformed ideas. Remove generic padding from `_parse_ideas_response`.
5. Preserve positional mappings explicitly in batched mode. In per-idea mode, attach each call to its exact idea before continuing.
6. Store feedback as a hashed `FeedbackBundle`; pass its ID and eligible parent IDs into generation.
7. Capture the usage and artifact metadata already available from `_call_llm` in the LLM-call ledger.
8. Record every attempt separately. If a candidate is corrected once, the corrected attempt must link to the failed attempt and correction call.
9. Default generation 1+ to a bounded edit of a selected parent. An unrestricted exploration is allowed only when declared and justified.
10. Validate declared parameter count and symbolic complexity. During the next controlled run, use a recommended initial budget of 6–10 tunable parameters; require an explicit justification for larger policies.

Tests:

- Exact structured ideation output produces the expected number of distinct `PolicyIdea` objects.
- Missing, duplicate, or malformed ideas fail; none are silently padded.
- Batched and per-idea modes preserve candidate-to-idea mapping.
- Generation-1 repair, simplification, crossover, and exploration parent-count rules are enforced.
- Invalid generation and invalid code validation still emit uniquely identified records.
- Shared ideation-call tokens are counted once.
- One-repair-per-candidate means one linked repair attempt, not an unbounded outer-loop retry.

Exit criterion: a mocked two-generation run produces schema-valid candidate records with complete candidate, feedback, idea, proposal, parent, attempt, and LLM-call links.

### Stage B — production objective and checkpoint correctness

Primary files:

- `lares/core/training_pipeline.py`
- `lares/fit/objectives.py`
- `lares/fit/fitter.py`
- policy base class or trusted execution wrapper

Changes:

1. Add `fit_policy_with_objective(policy, data, objective, ...)` for an already generated policy.
2. Keep `fit_with_objective(structure, ...)` as a thin compatibility wrapper if needed.
3. Route the live evolutionary pipeline through the objective-aware fitter.
4. Use `DETERMINISTIC_MSE` for the next baseline and evolution runs.
5. Remove trainable `log_std` or other scale-only parameters from optimizer parameter groups. Prefer a trusted wrapper/base helper that returns a fixed positive scale; recommended initial value is `0.1`, logged in the manifest.
6. Change generation prompts so new policy code does not define trainable action-scale parameters.
7. Fix fitter finalization so `train_loss_final` and `validation_loss_final` use the named objective, not the legacy `bc_objective(std_weight=0.01)`.
8. Define one comparable rollout score type matching `(success, progress)`. Do not force it to `float`.
9. Evaluate and store the zero-shot checkpoint at step 0.
10. Keep `initial_state`, `final_state`, and `selected_state` distinct.
11. After selecting and loading the best state, recompute all selected-state diagnostics: validation loss, per-axis losses, bounds, parameters, and any gate/term statistics.
12. Predeclare the checkpoint tie rule as: higher success, then higher progress, then earlier checkpoint.
13. Use a dedicated `probe` manifest/stream rather than overloading GIF environments.

Tests:

- A policy with a learnable `log_std` fixture leaves that parameter unchanged or excluded under deterministic fitting.
- The reported final loss equals a direct evaluation of the configured objective.
- A synthetic run whose best checkpoint is not the final step reports the best checkpoint's parameters and diagnostics.
- Step 0 can be selected.
- Tuple rollout scores compare correctly and deterministically.
- Exact score ties select the earlier checkpoint.
- Selected-state serialization and reload reproduce the selected policy's outputs.

Exit criterion: the live evolutionary fitter produces internally consistent selected and final reports, and no scale-only parameter changes under deterministic fitting.

### Stage C — records, archive, gates, and summaries

Primary files:

- `lares/search/records.py`
- `lares/search/archive.py`
- `lares/core/training_pipeline.py`
- `scripts/summarise_experiments.py`
- symbolic policy base class and validators

Changes:

1. Strengthen `ExperimentRecord.validate()` for every terminal status, including rejected candidates.
2. For generated candidates, require generation, slot, attempt, code hash, validation result, prompt/call references, and idea lineage even on failure.
3. For evaluated candidates, additionally require manifest/action mode, objective config, selected step/state, evaluation report, resource ledger, and reconstruction metadata.
4. Make record writing append-only or versioned. Do not silently overwrite material evidence.
5. Add a reconstruction command/test that loads a record, rebuilds its policy, loads its selected state, and compares actions on a fixed probe batch.
6. Reconcile run-level resource totals against unique call IDs and evaluation artifacts.
7. Upsert archive entries after expanded evaluation. Store `screen_metrics` and `confirmed_metrics` separately and use confirmed metrics for parent selection when present.
8. Update prompts and examples to assign gate overrides:

   ```python
   gate = self.record_gate("gate_name", gate)
   action = action_using(gate)
   ```

9. Extend the validator to force every advertised gate and assert that at least one action component changes on an applicable diagnostic state.
10. Change `summarise_experiments.py` so “traceability complete” requires lineage, feedback, prompt hashes, LLM usage, state artifacts, manifests, resource accounting, and successful reconstruction.
11. Compute validated candidates from `validation.passed`, not `attempted - invalid`.
12. Include successful-after-correction attempts and repair-call costs in repair summaries.

Tests:

- Rejected generated records cannot pass validation with empty lineage fields.
- An evaluated record missing selected state, prompt evidence, or resources cannot pass traceability.
- Reconstruction produces identical deterministic actions on a fixed input batch.
- Prefer bit-identical evaluation on a small fixed manifest; if the environment prevents bit identity, require a documented tolerance and matching success outcomes.
- Archive metrics change from screen to confirmed values after expanded evaluation.
- A no-op forceable gate fails validation; a correctly assigned gate passes.
- Resource totals equal the sum of unique artifacts/calls without duplicate shared ideation costs.

Exit criterion: summaries cannot make a stronger traceability or reliability claim than the underlying records support.

### Stage D — two-generation smoke test

Run a small end-to-end test after Stages A–C pass. Use the development partition only.

Recommended shape:

- 2 generations
- 3 requested candidates per generation
- deterministic mean action
- fixed action scale
- deterministic MSE fitting
- checkpoints `{0, 500, 1000, 1500, 2000}`
- isolated probe, screen, and expanded manifests
- normal feedback generation between generations
- at least one correction path exercised with a controlled malformed fixture if the live model does not naturally fail

Required smoke-test checks:

- [ ] All requested slots have unique candidate and attempt IDs.
- [ ] Every generation-0 candidate has a valid idea.
- [ ] Every generation-1 candidate has a valid proposal, idea, feedback ID, context parents, and valid semantic parents for its edit mode.
- [ ] Every LLM response and prompt is stored and hashed.
- [ ] Token totals reconcile using unique call IDs.
- [ ] No scale-only parameter changes during deterministic fitting.
- [ ] Step-0 and scheduled checkpoint probes are recorded.
- [ ] Selected parameters and diagnostics match the selected checkpoint.
- [ ] Simulator steps and gradient steps are reported separately.
- [ ] Expanded results update archive entries.
- [ ] At least one candidate from each generation reconstructs exactly from its record.
- [ ] Summary tooling reports no false “complete” fields.
- [ ] Neither the final test partition nor Qwen is accessed.

If any check fails, stop and fix it before running a scientific comparison.

### Stage E — fresh development-confirmation experiment

After the smoke test passes, create a new disjoint **development-confirmation** manifest. It must not overlap prior tuning cases and must not be the final test set.

Predeclare this first comparison:

| Arm | Objective | Selection |
|---|---|---|
| Fixed-2000 | Deterministic MSE | Always step 2000 |
| Rollout-selected | Deterministic MSE | Best of `{0, 500, 1000, 1500, 2000}` |
| Phase-balanced | Phase-balanced objective | Same checkpoint rule |

Controls:

- Same symbolic structures in every arm.
- Same fresh development-confirmation cases.
- At least five seeds per structure/arm.
- Same probe and evaluation budgets.
- Record paired per-structure, per-seed differences.
- Report mean, median, individual pairs, and an uncertainty interval suitable for the actual small sample; do not rely only on a normal `1.96 × SE` interval.

Decision rule:

- Promote an objective/checkpoint rule only if the improvement is consistent enough to justify its added selection cost and does not degrade robustness.
- If results are mixed, retain deterministic MSE with the simpler predeclared rule and document the uncertainty.
- Do not use the final test set to resolve the choice.

### Stage F — correction-LLM repair and clean ablation

Start this only after Stages A–E are complete.

The current ablation result should be relabeled as a pilot. Across its six structure/seed pairs, `optimizer_only - full` was `+0.35, +0.20, +0.05, +0.25, -0.05, 0.00`, with mean `+0.133`. The small-sample t interval is approximately `[-0.031, +0.298]`, so it supports a directional optimizer advantage, not a definitive significance claim.

Required repair path:

1. Create a numeric evidence packet from evaluation and diagnostic artifacts.
2. Ask a correction LLM for a schema-valid `MutationProposal` tied to explicit parent and feedback IDs.
3. Generate or patch the policy once.
4. Validate, refit with common preprocessing, and evaluate under an explicit matched budget.
5. Record the entire repair chain and its cost.

Recommended clean arms:

- no repair
- evidence-free LLM repair
- numeric-diagnostic LLM repair
- numeric + intervention LLM repair
- local parameter search
- global parameter search

Checkpoint selection and initial fitting are common preprocessing for every arm, not a special “optimizer-only” repair arm. Match or explicitly normalize LLM calls, gradient steps, expert-assisted episodes, repair evaluation episodes, and wall time. Add VLM evidence only after the numeric correction path works.

## 6. Historical provenance policy

Do not rewrite old raw records to imply evidence that was never captured. Produce a versioned provenance sidecar or migration output.

For the known generation-1 run:

- The feedback context can recover `context_parent_ids` for `gen0_cand1` and `gen0_cand2`.
- The transcript plus `Iter_1_ideas.json` can recover the positional idea mapping: `gen1_cand0 → idea0`, `gen1_cand1 → idea1`, `gen1_cand2 → idea2`.
- Do not invent semantic `parent_ids` or `proposal_id` if the model never declared them.
- Mark these records `lineage_status: partial_recovery` and list the fields that remain unknown.

The summary must distinguish:

- `complete`
- `partial_recovery`
- `legacy_missing`
- `failed_validation`

## 7. Minimum automated test matrix

| Layer | Required coverage |
|---|---|
| Unit | Strict idea/proposal parsing, lineage rules, stable IDs, objective dispatch, scale freezing, checkpoint tie rule, resource aggregation. |
| Component | Existing-policy fitting, selected-state diagnostics, record validation, reconstruction, archive upsert, gate intervention. |
| Mocked integration | Two generations with success, invalid generation, correction, refit, screen, expanded evaluation, and summary. |
| Live smoke | Two generations × three candidates on development data with real LLM calls and full provenance. |
| Scientific confirmation | Fresh disjoint development-confirmation manifest, paired arms, at least five seeds. |

All tests should be deterministic where possible and must pin manifests, seeds, action mode, environment mapping, objective config, and checkpoint schedule.

## 8. Definition of done

This debug cycle is complete only when:

- [ ] The live pipeline produces typed generated candidates rather than untracked parallel lists.
- [ ] Generation-1+ lineage is complete by construction.
- [ ] Malformed ideation cannot pass through padding.
- [ ] The production fitter uses the named objective and fixed scale.
- [ ] Checkpoint state, step, metrics, diagnostics, and serialized parameters agree.
- [ ] Evaluated records reconstruct their deterministic actions.
- [ ] Actual simulator usage is separated from optimization work.
- [ ] Archive parent metrics reflect expanded confirmation when available.
- [ ] Forced gates demonstrably affect actions.
- [ ] The two-generation smoke test passes every check.
- [ ] The fresh development-confirmation experiment is complete and reported without final-test access.
- [ ] The repair study uses a real correction-LLM path and matched preprocessing/budgets.

Only after these conditions hold should the project un-block Phase 7.

## 9. Immediate implementation checklist

The next coding session should do exactly this:

1. Add `FeedbackBundle`, `GeneratedCandidate`, and `LLMCallRecord` schemas plus validation tests.
2. Refactor `policy_generation.py` to return `GeneratedCandidate` objects and delete idea padding.
3. Thread feedback, parents, ideas, proposals, attempts, and LLM-call IDs through `training_pipeline.py` into records.
4. Refactor the objective-aware fitter to accept an existing generated policy.
5. Switch live evolution to deterministic MSE, freeze action scale, and correct final/selected reporting.
6. Add selected-state artifacts, resource-ledger fields, reconstruction, and stricter record validation.
7. Fix archive upsert and gate override semantics.
8. Update `summarise_experiments.py` and add the historical provenance sidecar.
9. Run the unit and mocked integration suites.
10. Run the two-generation live smoke test and stop on the first failed acceptance check.

The current evidence is sufficient to begin these changes; no additional code-space information is required before Stage A.
