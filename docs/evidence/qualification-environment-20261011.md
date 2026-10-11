# Qualification environment repair — 2026-10-11

Clean implementation **4f23993315554f68be3352d7d15533a0e445b260** keeps device
qualification artifacts outside source and resolves relative library arguments
before changing cwd. Earlier **dd5e220** preserves activated vendor PYTHONPATH
in fresh checkpoint/resident children. No graph,model,VJP or compiled source
changed. [Audited records](qualification-environment-20261011.json) retain exact
sources,receipt/log hashes and terminal service checks.

| Qualification | Result |
| --- | --- |
| dd5e220 related CPU entry/library checks | 46 passed,92.94s |
| 4f23993 related CPU entry/library checks | 50 passed,94.31s; no skips |
| 7da5242 component-entry lifecycle follow-up | 30 passed,0.60s; no skips; synthetic CPU checks |
| 4f23993 Python FP32 NPU mixed-pdg | All7 checks passed: complete observables,independent VJPs/None,chunk continuation,trace-disabled results,3 AdamW steps,fresh-process checkpoint,CPU checkpoint handoff |

The NPU case used independently computed schedules and original1e-6/1e-5 tensor
tolerances. Physical9 mapped to logical0; external occupancy was allowed under
explicit user authorization. Four CANN compiler-pool limits were set to1 in
launch configuration. Peak sampled aggregate RSS was10379898880B (9.67GiB),
below the unchanged12GiB budget; this sums shared mappings across processes.
The500s workload/600s service limits were unchanged. The frozen source stayed
clean; each completed service has MainPID0,empty cgroup and no guard survivors.

Earlier failures remain failures: batch03's CANN tbe import loss,batch04's RSS
cap,calibration01's dirty-source refusal,and batch05's launcher import before
any device selection. The9 vendor-generated files from batch04 were hashed and
relocated into its retained run directory; tracked source bytes never changed.
The site-only launcher cwd correction uses new batch06/child07 identities.
These are diagnosed configuration repairs,not performance retries.

This report qualifies the environment correction and one exact device case.
The full e565f96 CPU inventory and a926703 backend builds keep their prior
scopes. Full new-source NPU modules,training,resident/multicard and profiling
are separate work; no performance measurement or GPU execution passes here.
The current submitted batch and next actions belong only in [STATUS](../STATUS.md).

The follow-up **7da524268cc8b1eb677cbd99d2847f6690778ea4** makes direct component
profiling safe when launched from the checkout: device resolution,model setup,
measurement and export all use the artifact cwd, restored on success/failure.
An unavailable requested device leaves a failed record with no resolved device
or measurement. The30 clean CPU tests include synthetic initialization caches,
resolution/operation failures and trace export; they are not device or timing
evidence. AST source hashes prove make_case/operation unchanged from4f23993.
The active NPU batch uses that earlier fixed source and already supplies artifact
cwd; no live plan/source was edited and no measurement was repeated.
