# Immutable measurement-lifecycle qualification

Commit `103f5b6c8e9b2e6e46a7f2733185433de2cb5f3f` passed all **17**
dependency-free checks on its immutable checkout in 8.048 seconds. All 1588
tracked file hashes were checked afterward. See the [record](measurement-lifecycle-20261004.json).

Four new real-process regressions cover an exited adopted grandchild, natural
helper teardown, a live helper leak and a nonzero helper exit hidden by a
successful parent. The unchanged tests cover NUMA/CPU admission and placement,
real overlap, memory refusal, timeout cleanup and failed-companion cancellation.
A real fork reproducer on the old controller demonstrates the zombie
misclassification; it does not identify the helper in the retained NPU failure.

Successful completion now reaps exited descendants and checks their statuses.
Live descendants get at most two seconds to exit naturally within the unchanged
lane timeout. A live leak or nonzero descendant still fails and is cleaned up.
This is process-lifecycle evidence, not graph correctness or full-size parallel
performance qualification. The new full-size screen is recorded in STATUS.
