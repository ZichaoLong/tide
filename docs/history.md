# Explicit historical lookup

The active tree contains the current instance cycle and curated applicable lessons.
Historical implementation investigation is done only when the user explicitly asks
for it. Ordinary startup, optimization and documentation reading use current files.

The last complete tree before the 2026-11-18 governance retirement is:
`728d19a470d16d392d5dd6a2452161d1a3e9959e`.
It retains the original paths and bytes of retired contracts, plans and evidence.
The model rewrite retired its predecessor at
`9afe84773afb72214e80b615cc704667945f0ccb`.

For an explicitly requested historical item, use `git show COMMIT:PATH` or an
isolated temporary checkout. The pre-governance `docs/history-migration.md` and
`docs/history/migration-20261030.json` explain earlier rewritten commit identities.
Do not reinterpret a mapped source as newly verified evidence.

Old evidence retains its own source, environment and result identity in Git.
Artifact hashes identify former bytes; they do not promise raw binaries, traces
or checkpoints remain on disk. Current reproduction uses tracked generators,
configurations and the current [development entry](development.md).
