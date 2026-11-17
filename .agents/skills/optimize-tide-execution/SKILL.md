---
name: optimize-tide-execution
description: Optimize tensor flow in the current FC/HB models while preserving their selected mathematics.
---

# Optimize Tide instances

Optimize tensor flow in the current FC/HB models while preserving their selected mathematics.

Use the performance lessons in docs/performance-lessons.md as observations and methods, never as obligations to retain old code, examples or APIs. Inspect loops, shapes, copies and tensor lifetimes. Batch across B/L and compatible nodes; avoid per-event splitting/reassembly. Evaluate KV, reception and regional Attention separately: Flash/fusion, ordinary batched math and tiled kernels compete by actual shapes/devices. Preserve original-time windows and weighted gradients. Compare independent formulas, no-grad/grad-forward/full training, matched optimized Transformer/SSD and direct same-function implementations when useful. Record actual work and peak memory; parameter count alone is not work. Use finite additive/subtractive diagnosis, then adopt/redesign/defer with a reason. No compulsory duplicate implementation, full matrix or absolute speedup threshold.

User instructions override historical guidance. Choose serial work, parallel tools or agents by benefit; coordinate resource use and conflicting edits. Main agent integrates and commits. Long jobs use frozen source, background.slice and explicit time/memory bounds.
