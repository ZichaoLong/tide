# Background status and narrow build cleanup

At2026-10-04T16:29:30+08:00,the unattended manager remains active in its independent
background.slice service(MainPID958799). It waits for the eleven-device resident
FP16 Attention complete-training process(service749302,consumer974870),which has
run about64minutes and still has CPU activity. No terminal model result exists.
[Reviewed records](background-status-cleanup-20261004.json).

The preceding CPU streaming group ended exit1 at07:25:39.705740Z,with an empty
cgroup. Cell60(Settle/LibTorch/Add/CPU streaming) produced12288outputs,cut816 and
1188500candidate events;warmup533.699052s,measured773.663847s and separate
construction45.450142s. The consumer completed,but773.663847s exceeds its original
600s step allowance. The strict terminal audit therefore accepts zero cases and
retains the group as failed. Cells84/108 never started and remain eligible for
independent first execution;cell60 will not be retried by the queue. Formal FP32
coverage remains14/120;no new speed recommendation or equivalence claim follows.

The user authorized safe project cleanup. A reviewed dry run selected only
regenerable CMake dependency/build-rule files in eight obsolete development build
trees absent from current reviewed documents,plans and audits. Live project open
files/mappings and the queue's644hashed inputs were checked before deletion.
64,568files were removed,freeing929,161,216allocated bytes(0.865349GiB). The full
per-file path/inode/size inventory and removal receipt remain under
`TASK/plans/storage-cleanup-20261004-01/`. After deletion,all644queue input hashes
still match. Sources,objects,archives,shared libraries,binaries,configuration,
results,logs,profiles,qualified builds and protected historical work remain.
Rebuilding an affected old development tree requires regenerating its CMake files.

Data free space increased from173,598,003,200to174,527,111,168bytes during deletion;
root free space remained about12,158,955,520bytes(11.32GiB). Other workloads can
change free-space readings independently. The current queue's24GiB data/8GiB root
stop reserves are not reached. No SDK,environment,reference repository or other
user's data was cleaned. No running experiment was restarted or resubmitted.
