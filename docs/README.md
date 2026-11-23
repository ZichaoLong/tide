# Current documentation map

Choose the entry for the user's task. The read-only command `python scripts/status.py`
shows this checkout and its dated handoff. Add `--json` for programmatic use.
[Four task entries](../AGENTS.md#re-entry) · [How to read status](downstream-contract.md#read-only-status)
The current source is `src/tide`, tests `tests`, comparisons `benchmarks`, configurations
`configs`, and fixed mathematical source `math`.

| Question/change | Owning document(s) |
|---|---|
| Product route, selected cycle, architecture freedom | [project scope](project-scope.md) |
| Work in progress, blockers, jobs, next command | [STATUS](STATUS.md) |
| Remaining foundation work | [ROADMAP](ROADMAP.md) |
| Adopted mathematics / where implemented | [semantics](semantics.md), [semantic map](semantic-map.md), [architecture](architecture.md) |
| Public API, configuration, training, state | [instances](instances.md) |
| Reception self-Attention and optional parameter views | [Reception execution](reception-execution.md), [parameter indexing](parameter-indexing.md) |
| First backbone output, prompt readiness, cached decode and uninterrupted streams | [inference measurement](inference-benchmark.md) |
| Named training choices, explicit overrides and per-mode advice | [execution recommendations](execution-recommendations.md) |
| Arithmetic policy and recorded numerical scope | [precision](precision.md) |
| Inputs that exposed a numerical failure, independent answers and regression commands | [numerical failure modes](numerical-failure-modes.md), [pytest selection](../tests/numerical-regressions.txt) |
| Four joint matches, work counts and timing | [benchmark contract](benchmark-contract.md), [runnable comparisons](benchmarking.md) |
| FP32 throughput by scale/topology, active work and computation formulas | [computation and throughput](computation-and-throughput.md) |
| Environment, checks, durable execution | [development](development.md), [verification](verification.md), [execution](execution-flows.md) |
| Measured results and applicable experience | [evidence](evidence/README.md), [performance lessons](performance-lessons.md) |
| Downstream request, delivery, consumer acceptance and batch adoption | [collaboration](downstream-contract.md), [r24 case](../../tide-learning-lab/docs/requests/pilot-r3-r24-quad.md) |
| Artifact lifecycle and cleanup | [retention](artifact-retention.md) |

When a semantic/API/precision/timing contract changes, update its owning page and
relevant checks; record the actual verified result separately. Keep plans and
qualifications distinct. Skills link these contracts instead of duplicating them.

The working tree contains this cycle. Old APIs, matrices and evidence remain in Git;
[history](history.md) is for user-requested historical lookup, not ordinary startup.
