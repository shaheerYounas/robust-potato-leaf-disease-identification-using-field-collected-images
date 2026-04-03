# In Progress

This folder tracks the work that should receive attention right now.

## Current Focus

The project is in the transition from completed experimentation to final evaluation framing, report writing, and reproducible deliverables.

## Active Priorities

1. Sync Phase 2 outputs from notebooks/Drive into the local workspace.
2. Keep the full workflow organized inside the main notebook already in use.
3. Lock the final best-model path.
4. Define the final evaluation protocol clearly.
5. Start robustness experiments.
6. Start report writing in parallel.

## Current Technical Truth

- Strongest evidence lives in `Advance_Practice_Potato_Leaf.ipynb`
- All code-related project execution should remain in `Advance_Practice_Potato_Leaf.ipynb`
- Literature review research is complete, but the final written section is not yet drafted into the report
- EDA is already strong and mostly complete; only selective high-value additions may still help
- Benchmarking is strong in notebook evidence, including model comparison metrics
- Local workspace still lacks a complete final package of:
  - `benchmarking_results.csv`
  - classification reports
  - confusion matrices
  - training curves
  - EfficientNet and Hybrid checkpoints
  - deployment scripts

## Files Most Likely To Be Worked On Next

- `Advance_Practice_Potato_Leaf.ipynb`
- `Notebook/Advance_Practice_Phase_2.ipynb`
- `instructions/MILESTONES.md`
- `requirements.txt`

## Watch Outs

- Do not move dataset folders unless code paths are updated intentionally.
- Do not split the core modeling workflow across multiple training scripts; keep it in the main notebook unless the project rule changes.
- Do not treat notebook-only results as fully packaged deliverables.
- Do not describe the literature review as unfinished research; the research is done, but the writing is still pending.
- Keep final-report writing aligned with actual saved outputs, not just planned outputs.
