# Downstream collaboration for Tide instances

Current protocol: `tide-instances-collaboration-v2`. This replaces the old foundation
CLI/plugin protocol for new instance consumers. Existing requests and dependency
locks retain their original adopted revision; rewriting this page does not migrate
or accept any downstream installation.

Install a fixed `tide-instances` source/wheel in the consumer's private environment.
Start from [instances](instances.md), [development](development.md), and the checkout
`examples/train.py` mechanism example (the wheel installs `tide`). The public entry is `tide`, not a benchmark
script. Keep the source commit, package/wheel identity, math revision, complete model
configuration, actual device/dtype and environment with the experiment dependency lock.

| Responsibility | Owner |
|---|---|
| Input data, task head/loss, recipe, model quality | Experiment repository |
| Selected model semantics, modules, gradients and state mechanisms | This repository |
| Full application checkpoint (data/RNG/scheduler as needed) | Consumer, composing the model checkpoint API |
| Mathematical changes | Explicit upstream proposal and a newly adopted fixed version |
| General performance experience | Curated here, with applicable scope and measured evidence |

A request stays in its originating repository with a stable ID, fixed dependency,
actual configuration, observed problem, available evidence/reproducer, and acceptance
conditions. Unknowns can be stated; a full performance matrix is not required to ask.
Accepted foundation work enters ROADMAP and current activity STATUS. Delivery gives
implementation/evidence commits, API/state changes, exact verified scope and a
consumer reproduction command. Each consumer records its own acceptance before
changing its lock. Upstream delivery never implies downstream acceptance.

Freeze installations used by active experiments. Upgrade in a separate installation,
check affected inference/gradients/updates/state and then adopt explicitly. Source
identity, exact checkpoint data, tolerant numeric replay and bitwise optimizer replay
are separate claims. Public API redesign is allowed; expensive compatibility is
replaced by a documented migration when appropriate.

Repository access or a request file does not grant cross-repository write/push or
message-sending authority. Preserve downstream worktrees, installations and model
paths during foundation cleanup. Old protocol details are retrievable from their
adopted commits only when historical investigation is requested.
