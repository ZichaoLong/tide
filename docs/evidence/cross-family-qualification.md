# Cross-family clean qualification and audit

The immutable qualification source is
`eaa15c6f1d58291d818f8872ee20987abbb3d63b` at
`/var/tmp/zlong-graph-execution-foundation/qualification/cross-family-20260925`.
The durable unit `tide-cross-family-qualification-20260925-a` ran in
`background.slice` with Nice 10 and is inactive/dead.

The clean CPU gate passed **8,577/8,577 tests** for FP32 and FP64 on aarch64
with Torch/LibTorch 2.10.0+cpu. The same driver exported and rebuilt the source
in a separate directory, then ran **60/60 relocated foundation-v2 smoke
variants with 0 failures**. Standalone Settle binaries passed for both FP64 and
FP32. The older qualification attempt with 8,547 passes and 30 native-unpacked
replay-counter failures remains retained as historical failure evidence and was
not relabeled.

The read-only audit command was:

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
/home/zlong/anaconda3/bin/python \
artifacts/cross-family-drivers-20260924/review.py \
/var/tmp/zlong-graph-execution-foundation/qualification/cross-family-20260925 \
/var/tmp/zlong-graph-execution-foundation/artifacts/cross-family-qualification-20260925-a \
/var/tmp/zlong-graph-execution-foundation/artifacts/cross-family-performance-20260925-a
```

It wrote `artifacts/cross-family-performance-20260925-a/reviewed-audit.json`
with state `reviewed`. The audit checked 535 tracked source files against the
qualification archive and the relocated export, 13 binaries in each of the
original and relocated build directories, native binary identities, complete
run records and child reaping. It validated 252 records and made 567
observable comparisons; all were passed, with zero record or comparison
failures. It checked the 60 relocated smoke runs, all 180 medium runs and all
12 large records, including the retained timeout record.

This evidence establishes the declared CPU FP32/FP64 finite profile, clean
source/build/result identity and relocation behavior for E1–E5. It does not
extend support to CUDA, Ascend, another host architecture, arbitrary imported
models, higher-order AD or target-scale Settle execution beyond the bounded
assessment reported in [the performance evidence](cross-family-performance.md).
