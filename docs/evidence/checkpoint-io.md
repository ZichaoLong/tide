# Atomic exclusive checkpoint publication

Tested clean source: 00bbf78fb99e410d46b4d7e36e37266eb3d8cf9e.
CPU aarch64 Linux, Python 3.11.15, Torch/LibTorch 2.10.0+cpu, C++11 ABI;
FP64/FP32, one ATen/BLAS thread, two build jobs. Graph v13/checkpoint v5.
No serialized-format or load-semantics change.

Unit tide-foundation-checkpoint-io-qualified-20260922-0000 terminated
inactive/dead, MainPID 0, exit 0. **88 tests passed in 6.82s**. Both status.json
and development.json under artifacts/checkpoint-io-qualified-20260922-0000/
are passed; source dirty status is empty. Started 2026-09-22T00:00:01Z,
finished 00:00:12Z. The main checkout/build stayed frozen during this short gate;
concurrent full qualifications use separate worktrees and builds.

The exact command is retained in status.json: scripts/job.py wraps
scripts/develop.py --jobs 2, selecting test_checkpoint_io.py,
test_checkpoint_ownership.py, test_checkpoint.py and these resume IDs from
test_single_graph_resume.py::test_single_pdg_resume_matches_two_clock_updates:

- float64-cursor-adamw-True-hst-all-softmax
- float32-cursor-adamw-True-hst-all-softmax
- float64-native-serial-sgd-False-hard-add
- float32-native-serial-sgd-False-hard-add

The build source fingerprint and all binary hashes were refreshed before testing.
After exit every one of the 327 archived source files matched the unchanged tree.
Source archive SHA256:
a2bfa6bd9c3041fd079b8dfc7b97f425c358eb68c21a59fdaea7c885c9651b66.

## Failure and repair

At prior source d233429cd5807614869214dcea21d9492e309fd1, injected ENOSPC during
serialization left an 18-byte final checkpoint that failed weights-only loading.
The record and partial file remain in artifacts/checkpoint-partial-write-repro/.

Save now writes into a same-directory staging file, flushes/fsyncs its bytes,
creates the target with an atomic no-overwrite hard link, removes staging and
fsyncs the directory. Tests inject serialization/file-fsync failures and verify
no published target, staging cleanup and successful retry. Existing and racing
writers are preserved. A serialization callback checks target absence while
writing. A directory-fsync error after publication remains an error, with a
complete loadable target; retry does not overwrite it.

The remaining 78 tests check named ownership, malformed optimizer state,
round-trip values and four full single-PDG save/restore/update trajectories.
Native serial/packed execution is exercised through the Python checkpoint
controller. This is not a standalone C++ writer/optimizer format, a composite
two-clock application checkpoint, a power-loss test or broader hardware evidence.
A process killed before cleanup can leave an orphan staging file; see
../checkpoint-ownership.md for the exact filesystem and post-publication contract.
