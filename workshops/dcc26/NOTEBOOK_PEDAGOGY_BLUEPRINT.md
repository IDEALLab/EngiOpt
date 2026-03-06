# DCC26 Notebook Pedagogy Blueprint (Pre-write Source)

This document is the canonical pre-write for workshop notebooks. Notebooks should be generated from this structure, not authored directly as raw `.ipynb` first.

## Teaching Design Principles

1. Every technical step is paired with a markdown teaching cell.
2. Every code section has local context: why, inputs, outputs, checks, failure modes.
3. Benchmark science is explicit: objective, feasibility, diversity, novelty, reproducibility.
4. Discussion prompts are embedded and mapped to workshop breakout questions.
5. Participant and solution tracks share the same pedagogical arc; only implementation detail differs.

## Common Cell Pattern

For each section:

- Purpose: why this step matters for benchmark credibility
- Inputs: what artifacts/variables are required
- Action: code operation performed
- Success check: what output indicates correctness
- Failure modes: common pitfalls and fixes
- Discussion bridge: one reflection question

---

## Notebook 00: Setup + API Warmup

### Learning objective
Understand EngiBench benchmark contract components and reproducibility controls.

### Section plan
1. Read-me-first + copy mode + runtime expectation
2. Concept cell: EngiBench vs model libraries
3. Environment bootstrap
4. Reproducibility cell (seed, versions)
5. Problem instantiation (`Beams2D`) + inspection
6. Dataset inspection and shape sanity
7. Render one sample and explain representation
8. Explicit constraint violation check with interpretation
9. Reflection prompts tied to comparability across papers

### Discussion trigger
Which benchmark settings must be fixed for fair method comparison?

---

## Notebook 01: Train + Generate

### Learning objective
Implement an EngiOpt model against EngiBench data while preserving evaluation-ready artifacts.

### Section plan
1. Read-me-first + copy mode + expected runtime
2. Concept cell: inverse design framing, conditional generation assumptions
3. Bootstrap deps and imports
4. Configuration and artifact contract
5. Data subset construction and rationale (runtime vs fidelity)
6. Model definition and optimizer
7. Training loop with diagnostics + expected loss behavior
8. Generation from test conditions
9. Quick feasibility precheck (not final evaluation)
10. Artifact export contract (npy/json/checkpoint/history/curve)
11. Optional W&B logging: train curve, scalar logs, artifact bundle
12. Visual sanity grid
13. Discussion prompt: training loss vs engineering validity mismatch

### Discussion trigger
Can lower train reconstruction loss worsen simulator objective or feasibility?

---

## Notebook 02: Evaluate + Metrics

### Learning objective
Run robust benchmark evaluation and interpret trade-offs beyond objective score.

### Section plan
1. Read-me-first + copy mode + expected runtime
2. Concept cell: why objective-only reporting is incomplete
3. Bootstrap deps and imports
4. Artifact loading strategy (local -> optional W&B -> local auto-build)
5. Per-sample evaluation loop (constraint + simulate)
6. Metric layer:
   - objective means and gap
   - improvement rate
   - feasibility/violation rates
   - diversity proxy
   - novelty-to-train proxy
7. Export layer: CSV + histogram + scatter + grid
8. Optional W&B evaluation logging (table + images + summary)
9. Interpretation rubric with examples
10. Breakout prompts mapped to workshop proposal

### Discussion trigger
Which missing metric would change conclusions for your domain?

---

## Notebook 03: Add New Problem Scaffold

### Learning objective
Understand minimal interface required for a reusable EngiBench-style benchmark problem.

### Section plan
1. Read-me-first + copy mode
2. Concept cell: benchmark-ready problem checklist
3. Scaffold imports and abstract contract explanation
4. Minimal `Problem` implementation skeleton
5. Toy simulator and constraints
6. Registration/discovery and deterministic behavior
7. Contribution checklist for real domains
8. Reflection prompts on leakage, units, and reproducibility metadata

### Discussion trigger
What metadata is minimally required so another lab can reproduce your new benchmark?

---

## Participant vs Solution Policy

- Participant notebooks: keep code TODOs, but each TODO has explicit completion checks and expected outputs.
- Solution notebooks: complete implementations plus concise inline comments for non-obvious logic only.
- Both tracks: keep identical markdown structure for pedagogical alignment.

## Quality Gate Before Publishing

1. All code cells compile.
2. Solution Notebook 01+02 execute end-to-end in workshop env.
3. Artifact contract is consistent between Notebook 01 and 02.
4. Copy-safe links use `#copy=true`.
5. Standalone readability check: each notebook understandable without live lecture.
