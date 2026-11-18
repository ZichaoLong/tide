# Historical preservation before authorized cleanup

Temporary historical commit material, removed from the final active tree.
The inventory is the reviewed dry run: 43 project-only roots,131 experimental
worktrees; main/downstream worktrees and model links are protected.
All candidate HEADs are ancestors of the baseline recorded in the inventory.

`dirty-worktrees.jsonl.gz` stores 27 original statuses, binary Git patches and
untracked bytes/symlinks. Reapply only in an isolated checkout of the recorded HEAD
when the user explicitly requests history. Bytes use base64 and hashes.
`small-records.jsonl.gz` stores unique source/small record bytes, duplicate/Git blob
identities, log tails, and explicitly marked JSON summaries. 170 records >4MiB
were not byte-retained; their paths/sizes are in inventory. All of those are JSON,
not unique source files. Summaries/hashes do not promise original numerical data.

The scripts are frozen one-shot tools, not a new default cleanup command.
Failed service identities remain in inventory even when their terminal transient
registrations are released. Neither retirement nor process exit repairs old failures.
