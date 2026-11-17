# Current handoff — instance rewrite in progress, 2026-11-17

User authorized implementation of the aligned rewrite plan. Fixed mathematics:
4f5b9f20a4030d68ee66ae2d26e34659a5e37234. Authorized branch:
graph-execution-foundation. Old teaching bindings and LH-DAG/PDG excluded.
Semantic correctness first, performance second; full mathematical traces,
universal decoding and historical backend/API matrices are not requirements.

Retired tracked implementation/test/build files removed without consulting their
contents. User's four untracked handoff files preserved. New source is src/tide,
new tests are tests. Complete fixed upstream text snapshot is math/.

Current work: new parameter/layout helpers, FC/KV, independent allocation and
Attention, SSD block math, and FC/HB graph generators. Read-only upstream and
public framework sources are allowed; retired foundation source/tests are not.
No new model qualification or performance result yet. No active durable jobs.
Agents own distinct new files; main agent integrates/tests/commits and resources.

Next: finish the first new CPU FC training/continuation loop, expand all declared
main candidates and HB, then finite paired CPU/NPU measurements. Do not restart
old experiment queues or old P1–P6. Shared device use remains under the explicit
continuing conversation authorization, with contention recorded.
