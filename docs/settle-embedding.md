# SettleGraph encoding and independent anchors

`SettleGraph(graph, ranks)` requires a distinct positive integer rank per region
and edge delay equal to target rank minus source rank. `stride=max(rank)+2`;
the output boundary rank is `max(rank)+1`. Each (sample,position) input is sent
through a stateless source adapter at `stride*position`; the output adapter
aggregates source-tagged terminal values. Adapters have no trainable parameters.

The direct Python executor settles a whole sequence by region rank, using exact
state/Full blocks. Its schedule does not call the TimedDAG planner. The native
SettleGraph execution path compiles the same encoding and runs the C++ frontier;
its ordinary dependency rules automatically form region sequence blocks. Graph
compilation is currently a Python frontend; the runtime and kernels are C++.
Native streaming provides a separate schedule for this encoding.

`project` removes boundary coordinates, restores external source tags, and
compares original event tensors, routes, states, histories, internal messages,
outputs and VJPs. Projection requires a complete position boundary, where no
messages remain in flight. Keep encoded continuation between native windows;
do not discard partially processed boundary state/messages. Logical time remains
the encoded time, and external position remains the original token position.

Independent topology anchors: Python/C++ `self_loop` for PositiveDelayGraph and
`chain` for TimedDAG; Python `settle_chain` for SettleGraph. They reject incompatible
graphs and own their propagation loops. They may share local formulas, and the
native specializations share the local region-block evaluator; neither invokes
a generic scheduler. Analytical cases separately check operator formulas.

The native standalone executable `tidegraph-smoke` runs a real delayed graph and
input VJP without Python. Its CLI runtime was adapted from the local
develop-portable-torch C++ asset; only CPU FP64/FP32 are accepted by this target.
