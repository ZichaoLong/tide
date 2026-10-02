# Exact C++ parameter initialization without int64 tensor temporaries

Implementation `be380db7451742c3fe50b57cb02263e81c64a4d6` passed fixed-source
consumer qualification and a separate CPU initializer comparison.
[Audited builds, tests and all timing samples](exact-initializer-20261002.json).

The C++ consumer composes the three affine LCG steps modulo 2^31-1 and fills
only the final CPU FP32 array, using the existing ATen thread budget. Parameter
names, seed, index mapping and payload casts are unchanged. Python retains the
independent three-step tensor definition. Core graph, scheduling, NPU libraries
and measured complete-step boundaries are unchanged.

Both CPU and NPU clients were freshly linked against byte-verified qualified
libraries, reusing only source/header/compiler-option-identical objects. CPU
qualification passed 25 tests: 380 independent scalar seed/modulo-boundary cases
and 24 actual standalone FP32/FP64 consumer comparisons. NPU qualification passed
30 standalone consumer cases: 12 resident complete-training, 12 resident inference
and six mixed training. These cover three families, Add/Attention, both schedules
and FP32/FP16 resident payloads. All requested tests passed without skips or
numerical tolerance changes. Unchanged Python/native-adapter cases were explicitly
deselected; they are not counted as new qualification.

The timing probe links the actual new model.cpp object and compiles the unchanged
ATen generator from prior source fe8a08b. Each of three fresh aarch64 CPU processes
uses one warmup and three alternating old/new measurements per shape, ATen/BLAS1.
Full-array memcmp runs outside the timers after every pair, including warmup;
all bytes match. Values are medians of the three process medians.

| FP32 source array | Elements | Old seconds | New seconds | Generation speedup |
| --- | ---: | ---: | ---: | ---: |
| 2048×2048 edge projection | 4,194,304 | 0.092060 | 0.014736 | 6.247× |
| 2048×6144 QKV | 12,582,912 | 0.287816 | 0.044218 | 6.509× |
| 50304×2048 output head | 103,022,592 | 2.300079 | 0.362125 | 6.352× |

These ratios concern parameter generation alone. They do not measure full-model
construction, graph scheduling, NPU kernels or complete-step throughput. No
full-size speedup is inferred from them. The original-wide Attention construction
observation motivated this change but was not rerun for a causal comparison.

Retained auxiliary failures: the first timing probe omitted the packet.cpp object
and failed at link before measurement; the corrected probe links that unchanged
dependency. The dependent first Add job then exited before NPU allocation. Neither
failure was a library correctness failure or a passing measurement. CUDA and
other host/software executions remain target-machine work.
