# Repository guidance

## Semantic documents

- Write semantic specifications for a reader who has not followed the design discussion. Define every symbol and concept before relying on it.
- Specify observable meaning: inputs, outputs, shapes, equations, state timing, topology constraints, invariants, and allowed variants. Do not prescribe a class, wrapper, or module name when it adds no independent semantics.
- Give a recurring concept a name only when the name helps distinguish a real role or operation. Define configurable operations next to their formulas. Remove one-off aliases and implementation-shaped CamelCase names that merely suggest future code organization.
- Reserve backticks for literal code identifiers, configuration values, filenames, and commands. Write conceptual roles as ordinary prose or mathematical symbols unless an exact serialized name is part of the experiment contract.
- Prefer equations for deterministic data transformations. Use short pseudocode only when ordering, conditional execution, state commit, or barriers would be harder to understand from equations alone.
- Introduce the intuitive data flow before detailed variants. Keep notation consistent across the base model, the single-layer example, HB-Lattice, losses, and experiment records.
- Keep the distinction between specification and implementation explicit. Never imply that a documented abstraction exists in software without checking the repository; put proposed software organization in an implementation plan rather than the semantic authority document.
- Avoid edit-history narration. State the current rule directly and keep qualifications close to the rule they constrain.

## Working on long documents

- Read and edit long documents in coherent sections, then perform a separate whole-document pass for terminology, definition order, cross-references, numbering, duplicated explanations, and Markdown rendering.
- Preserve useful detail, but remove repeated definitions. Later sections should instantiate or specialize earlier semantics rather than silently introduce a second version.
- After editing, search globally for removed names, undefined symbols, stale section references, and claims about existing implementation.

## Long-running commands

- For an authorized test, build, benchmark, or evaluation that should survive the current agent process, use the account's `run-durable-user-task` skill when it is available. A verified background submission may outlive the current response; wait for completion only when the user explicitly asks to wait, monitor, or finish.
- Launch from a stable source snapshot or keep the relevant checkout read-only until the task ends. Record the unit, working directory, exact command, source identity, logs and durable status paths, and inspection and stop commands. Never report submission itself as a successful result.
