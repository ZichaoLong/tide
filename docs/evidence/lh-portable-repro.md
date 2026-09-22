# Portable original-LH Attention reproduction kit

The user requested a clean-tree reproduction of the measured 17.27B Attention
configuration on the original Intel machine, with optional diagnostic output.
The source-only kit is implemented and has passed an independent local build
and four small forward checks. Target Intel execution remains unverified.

## Source, archive and clean qualification

Source: `eda5357da86ea0022c8e8aab43f75e315f71de79`. All 381 tracked files in the
frozen checkout match its Git archive, and git status is clean. LH numerical
sources are exact `a10fdb1883fccd63ec21e36cc9cffa294c63c9e2`.

Archive: `artifacts/lh-repro-export-20260922-0930/lh-a10-wide-repro-kit.tar.gz`,
179,312 bytes. SHA256:
`2fbf82e55a675fe372c1dbdcd9a7294a26787e7c4a0dd0e5f1c889a4ce7e7182`.
The archive has a manifest, file checksums and third-party notices. It contains
source/helpers, JSON headers and the retained original graph, without binaries
or model weights. The setup and use commands are in the
[kit README](../../tools/lh_repro/README.md).

The fixed graph has 232 static nodes and edge counts 984/984/232/8 across the
four blocks. Only graph_node_info_dir changes from its old absolute path to
`../test/graph-data`; graph numerical bytes remain unchanged. No graph generator
or LH Python interpreter is needed on the receiving machine. Python helpers
use only its standard library.

Unit `tide-lh-repro-check-20260922-0930` ran from
`/var/tmp/zlong-graph-execution-foundation/qualification/lh-repro-20260922-0930`
in background.slice, Nice 10, with two build/runtime threads on CPUs 156/157.
These CPUs are disjoint from the active large-LH run's 160–319; the host remains
shared. The unit finished 2026-09-22T09:32:42Z: persistent status passed/exit 0,
systemd inactive/dead, MainPID 0, Result=success.

Records: `artifacts/lh-repro-check-20260922-0930/` with 26 successful stages,
status.json, task.log, qualification.json, per-stage commands/logs, source audits,
binary hashes and four case reports. The private frozen launcher is retained at
`artifacts/lh-repro-helper-20260922-0930/qualify.py`. The native build took 169.30 s.
The exported archive was rehashed afterward and is unchanged.

## Verified scope

- Extracted the delivered archive into a new directory containing spaces and
  verified all checksums. Cloned LH into fresh, independent smoke/wide trees;
  checked out the exact revision. Original reference worktrees were unchanged.
- Applied the configurator to the fresh trees. It refused reconfiguration of
  the now-dirty trees without altering their configuration records.
- Default wide JSON hash is exactly
  `00a35b1704e8f8b79cfc11b7e01db157e1451544b4b8a642473513c96dd76c6a`, matching
  the measured 17,269,426,339-parameter Attention case. The default original
  test loop remains batch 512 / selectnum 1 / 100 steps. No large model ran during
  this packaging qualification.
- Rebuilt the original library and all four executables with GCC 10.3.1,
  Torch 2.10.0+cpu, C++11 ABI 1, Linux aarch64. All loader dependencies resolved.
- Original/diagnostic × nograd/grad-forward: each completed four tokens at
  width 64 / batch 4, with 23,133,347 parameters and native exit 0. The diagnostic
  child-module counts plus root counts equal the total; graph edge inventory,
  parameter CPU/FP32 checks, actual grad flags and positive RSS observations
  passed. All diagnostic logits have requires_grad matching the requested mode.
- Explicit unsupported device and dtype are rejected before model construction.
- Six focused summary checks pass: original timer units, preservation of the
  cold first step, malformed token/timer pairing, wrong parameter counts,
  incomplete logs and nonzero native exits. A complete log cannot override a
  failed process exit.

## Meaning and limits

Diagnostics print LibTorch/compiler/BLAS/OpenMP information, parameter groups,
graph counts, batch/width/configuration, grad mode, actual thread/affinity data,
RSS and an additional outer think timer. Diagnostic output and resource reads
are outside the original Think interval, but can still affect cache/scheduling;
the original executable variants remain available to measure that effect.
The summary's primary time uses original integer-ms Think/token pairs.
RSS is the process peak so far when printed, before final model destruction.

The model/input RNG remains the original unseeded behavior. These four small
runs are build, path and runtime/metadata checks, not strict same-weight output
parity, large performance evidence, backward correctness or optimizer tests.
Grad-forward performs no backward call. No compatibility claim is made for an
unexecuted Intel/LibTorch combination; the package must be rebuilt there.
