# DCC 2026 EngiBench / EngiOpt Slide Storyboard

Final deck: `workshops/dcc26/slides/benchmarking-ai-for-engineering-design-dcc26.pptx`

Contact sheet: `workshops/dcc26/slides/contact-sheet.png`

Presenters: Matthew Keeler, Soheyl Massoudi, Mark Fuge.

Core message: EngiBench and EngiOpt help engineering design ML become cumulative by turning design problems into executable benchmark contracts. Participants should leave less impressed by isolated generated shapes and more confident asking what was actually benchmarked.

Audience: DCC 2026 workshop participants: design computing researchers, engineering design researchers, and AI-for-design practitioners. Assume mixed ML depth and keep the simple notebooks as the main path.

## Red Line

1. Engineering design ML comparison is fragile when papers quietly change the task.
2. A benchmark is not just a dataset or leaderboard; it is an executable contract.
3. The contract makes hidden choices explicit: design space, conditions, objectives, constraints, dataset, renderer, simulator, and optimizer.
4. The simple notebooks let participants use that contract to frame a problem, train a generator, evaluate designs, and sketch their own domain problem.
5. The closing psychological goal is transfer: participants should know how this infrastructure can make their own research more reproducible, comparable, and reusable.

## Design System

- 16:9 widescreen PowerPoint.
- Off-white background, ETH-like blue section accents, near-black text.
- Red/orange accents for constraint and validity failure.
- One claim per slide.
- Large proof objects: rendered designs, training curves, metric summaries, warm-start curves.
- Calm footer with section and workshop label.

## Final Slide Sequence

1. **Title**: Benchmarking AI for Engineering Design. Includes presenters and ETH logo.
2. **Why this workshop exists**: Engineering design ML cannot become cumulative if every paper quietly changes the problem.
3. **What you should leave believing**: A benchmark is not just a leaderboard; it makes design claims reproducible.
4. **What you will do today**: Four simple notebooks turn that motivation into practice.
5. **Why benchmark before modeling?**: Same-looking tasks, designs, and scores can hide different engineering problems.
6. **Benchmark mental model**: A benchmark records the design task, not only the dataset.
7. **Notebook 00 bridge**: Frame your design problem before touching a model.
8. **Notebook 00 anchor**: Beams2D as the first concrete engineering task.
9. **API slide**: EngiBench turns design-problem questions into Python calls.
10. **Notebook 00 noticing slide**: Render, check validity, simulate, optimize.
11. **Notebook 01 bridge**: Train a generative model against optimizer answers.
12. **Training slide**: Training loss is a learning signal, not an engineering verdict.
13. **Generated vs baseline slide**: Look at designs, but do not stop at the eye test.
14. **Notebook 02 bridge**: Evaluate generated designs as engineering candidates.
15. **Evidence dashboard**: Feasibility, baseline feasibility, and generator wins.
16. **Failure mode**: Visual plausibility did not imply feasibility or performance.
17. **Warm-starting**: A generator may still be useful if it helps optimization.
18. **Discussion prompt**: Which metric would change the conclusion in your domain?
19. **Notebook 03 bridge**: Write your own design problem behind the same interface.
20. **Notebook 03 anchor**: Tiny cantilever problem with real engineering checks.
21. **Implementation checklist**: Explicit promises needed for a reusable problem.
22. **Domain translation worksheet**: Six prompts for participant research domains.
23. **Workshop logistics**: Use the simple notebooks; keep solution notebooks as facilitator references.
24. **Runtime fallback**: What to do if setup, training, optimization, or widgets are slow.
25. **Resources**: Simple notebooks, EngiBench docs, code, and paper.
26. **Closing**: Leave with a way to make your design problem executable, comparable, and reusable.

## Main Local Sources

- `EngiBench/docs/_static/img/engibench_problems.png`
- `EngiBench/docs/_static/img/problems/beams2d.png`
- `EngiBench/docs/_static/img/problems/airfoil.png`
- `EngiBench/docs/_static/img/problems/heatconduction2d.png`
- `EngiOpt/workshops/dcc26/simple/00_framing_your_design_problem.ipynb`
- `EngiOpt/workshops/dcc26/simple/01_training_a_generative_model.ipynb`
- `EngiOpt/workshops/dcc26/simple/02_evaluating_your_generated_designs.ipynb`
- `EngiOpt/workshops/dcc26/simple/03_writing_your_own_problem.ipynb`
- `EngiOpt/workshops/dcc26/artifacts/design_grid.png`
- `EngiOpt/workshops/dcc26/artifacts/objective_scatter.png`
- `EngiOpt/workshops/dcc26/artifacts/objective_histogram.png`
- `EngiOpt/workshops/dcc26/artifacts/training_curve.png`
- `EngiOpt/workshops/dcc26/artifacts/metrics_summary.csv`
