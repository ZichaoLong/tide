# Original-size resident FP16 copy failure

The eight-device Add inference attempt `formal-resident-fp16-add-inference02` failed before completing a step. `aclnnInplaceCopy` returned CANN361001 while constructing the resident execution. No FP16 throughput sample is available. The terminal audit retains the failed parent,consumer,monitor and released device lease. [Reviewed records](resident-fp16-copy-failure-20261004.json).

The workload remains clean `e69b3bd`,originalD2048/B512/T12/V50304,9,468,053,696 parameters and requested physical rows4. Plan02 keeps the900s step/2200s child bounds,FP16 payload and FP32 loss/adjoints/masters. Static eight-card admission estimated23.0107GiB/card;that estimate is not proof of runtime execution. The monitor ended after88.009s,consumer exit2,parent exit1,and the unit has an empty cgroup.

The native process's CANN error log reports a missing function handle for a cached float32 `TensorMove` kernel. All four installed TensorMove binary hashes match their metadata,and their declared kernel symbols exist. This narrows the next investigation to the actual operator/context sequence;it does not yet distinguish backend integration from a vendor-runtime issue. No shared toolchain was modified and no full-size retry was submitted.

A small reproducer links the already qualified control library and exercises captured copies across payload/control dtypes,scalar and wide shapes,and one/two device contexts. It is diagnosis,not a replacement for the failed full-size case or the passing small-model semantic gates. Subsequent work must retain this attempt and use fresh records after a demonstrated fix.
