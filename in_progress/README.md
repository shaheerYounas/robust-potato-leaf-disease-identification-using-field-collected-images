# In Progress

This folder tracks the work that should receive attention right now.

## Current Focus

The project is in the transition from completed experimentation to final evaluation framing, report writing, and reproducible deliverables.

## Active Priorities

1. Reuse the already saved local benchmark artifacts as the current benchmark source of truth.
2. Keep the full workflow organized inside the main notebook already in use.
3. Lock the final best-model path using the saved local benchmark CSV.
4. Define the final evaluation protocol clearly.
5. Run notebook Sections 7 to 11 in the intended GPU environment.
6. Continue report writing in parallel.

## Current Technical Truth

- Strongest evidence lives in `Advance_Practice_Potato_Leaf.ipynb`
- All code-related project execution should remain in `Advance_Practice_Potato_Leaf.ipynb`
- Literature review research is complete, but the final written section is not yet drafted into the report
- EDA is already strong and mostly complete; only selective high-value additions may still help
- Benchmarking is strong in local saved artifacts as well as notebook logic
- The local workspace now includes:
  - `artifacts/phase_2_benchmarking/metrics/benchmarking_results.csv`
  - classification reports
  - confusion matrices
  - training curves
  - local EfficientNet and Hybrid checkpoints
- The main remaining execution gaps are:
  - final evaluation protocol outputs
  - robustness outputs
  - Grad-CAM outputs
  - deployment/export outputs
  - final report integration

## Files Most Likely To Be Worked On Next

- `Advance_Practice_Potato_Leaf.ipynb`
- `Notebook/Advance_Practice_Phase_2.ipynb`
- `instructions/MILESTONES.md`
- `requirements.txt`

## Watch Outs

- Do not move dataset folders unless code paths are updated intentionally.
- Do not split the core modeling workflow across multiple training scripts; keep it in the main notebook unless the project rule changes.
- Do not treat prewritten notebook sections as completed evidence until their outputs are actually generated and saved.
- Do not describe the literature review as unfinished research; the research is done, but the writing is still pending.
- Keep final-report writing aligned with actual saved outputs, not just planned outputs.
