"""V2 clients: explicit policy groups, independent schedules and clock transitions."""
import torch
import _tide_native as core
from tidegraph.records import Continuation, Result
from tidegraph.native import Native
from tidegraph.native_records import from_continuation, to_continuation, window_records
from tidegraph.reference import run as reference
from tidegraph.streaming import run as streaming
from tidegraph.frontier import run as frontier
from tidegraph.settle import run as settle
from tidegraph.specialized import settle_layered as scalar_layered
from tidegraph import specialized_blocks
from foundation_workloads import initialize, inputs_for
from foundation_policy import resolve


class Execution:
    def __init__(self, config, variant, workers=4, trace=False):
        self.config, self.variant, self.trace = config, variant, trace
        self.policy = resolve(config, variant, workers, config.get("execution_options"))
        self.algorithm = self.policy["algorithm"]
        resolved = self.policy["resolved"]
        self.graph, self.spec, self.model = initialize(config, resolved["model"]["projection_layout"])
        self.values, self.external, self.stride, self.lengths = inputs_for(config, self.graph)
        self.options = dict(resolved["schedule"], mode="hst" if config.get("training_window") else "hard", trace=trace)
        self.native_options = dict(self.options, **resolved["kernels"])
        self.python_options = {k: self.options[k] for k in ("packed", "prefill", "full_autograd", "aggregate_autograd", "mode")}
        self.engine = self.stream_engine = self.compiled = None
        self.stream_algorithm = None
        transition = config.get("execution_pattern") in {"prefill-stream", "streaming"}
        self.eg = self.em = None
        if self.algorithm.startswith("native-settle"):
            body = Native(self.graph, self.model, **self.native_options)
            self.compiled = core.SettleGraph(body.compiled, self.spec.ranks)
            algo = "streaming" if self.algorithm.endswith("stream") else "frontier"
            self.engine = core.SettleExecutor(self.compiled, body.weights, body.options, algo)
            if transition and algo != "streaming":
                options = body.options; options.prefill = False
                self.stream_engine = core.SettleExecutor(self.compiled, body.weights, options, "streaming")
                self.stream_algorithm = "native-settle-stream"
        elif self.algorithm.startswith("encoded") or (self.spec and self.algorithm.startswith("python-stream")):
            self.eg, self.em = self.spec.embed(self.model)
            if self.algorithm.startswith("encoded"):
                algo = "streaming" if self.algorithm.endswith("stream") else "frontier"
                self.engine = Native(self.eg, self.em, algorithm=algo, **self.native_options)
                if transition and algo != "streaming":
                    self.stream_engine = Native(self.eg, self.em, algorithm="streaming", **dict(self.native_options, prefill=False))
                    self.stream_algorithm = "encoded-stream"
        elif self.algorithm.startswith("native"):
            algo = self.algorithm.removeprefix("native-")
            self.engine = Native(self.graph, self.model, algorithm="streaming" if algo == "stream" else algo, **self.native_options)
            if transition and algo != "stream":
                schedule = algo if algo in {"chain", "diamond"} else "streaming"
                self.stream_engine = Native(self.graph, self.model, algorithm=schedule, **dict(self.native_options, prefill=False))
                self.stream_algorithm = "native-"+("stream" if schedule == "streaming" else schedule)
        self.reset()

    def reset(self):
        self.q = Continuation(self.graph.identity, self.config["batch"])
        self.position = 0
        if self.compiled:
            self.eq = core.Continuation()
            self.eq.identity = self.compiled.encoded_graph.identity
            self.eq.batch_size = self.q.batch_size
        elif self.eg:
            self.eq = self.spec.embed_initial(self.q, self.eg)

    def _records(self, encoded):
        body = self.compiled.project(encoded)
        return Result(from_continuation(self.graph, body.continuation), *window_records(body))

    def advance(self, stop, phase="default"):
        if not self.position <= stop <= self.config["sequence"]:
            raise ValueError("invalid explicit position cut")
        streaming_phase = phase == "streaming"
        engine = self.stream_engine if streaming_phase and self.stream_engine else self.engine
        self.last_execution = dict(algorithm=self.stream_algorithm if streaming_phase and self.stream_engine else self.algorithm,
                                   prefill=self.options["prefill"] and not streaming_phase)
        py_options = dict(self.python_options)
        if streaming_phase:
            py_options["prefill"] = False
        if self.spec:
            x = self.values[:, self.position:stop]
            if self.compiled:
                encoded = engine.run(self.eq, x)
                self.eq = encoded.continuation
                result = self._records(encoded)
                result.stats = dict(encoded.stats)
            elif self.eg:
                end = stop*self.spec.stride
                xs = self.spec.external(x, self.position, encoded=True)
                if engine:
                    encoded = engine.run(self.eq, xs, end, sealed_until=end)
                else:
                    fn = reference if self.algorithm == "python-stream" else streaming
                    opts = {"mode": self.options["mode"]} if fn is reference else {k: v for k, v in py_options.items() if k != "prefill"}
                    encoded = fn(self.eg, self.em, self.eq, xs, end, sealed_until=end, trace=self.trace, **opts)
                self.eq = encoded.continuation
                result = self.spec.project(encoded)
            else:
                if self.algorithm == "python-layered":
                    result = scalar_layered(self.spec, self.model, self.q, x, mode=self.options["mode"])
                else:
                    fn = {"python-layered-block": specialized_blocks.settle_layered,
                          "python-chain-block": specialized_blocks.settle_chain,
                          "python-settle": settle}[self.algorithm]
                    # A single complete Settle position is a legal streaming block.
                    result = fn(self.spec, self.model, self.q, x, **py_options)
        else:
            end = stop*self.stride
            xs = [x for x in self.external if self.q.cut <= x.time < end]
            if engine:
                result = engine.run(self.q, xs, end, sealed_until=end)
            elif self.algorithm == "python-stream":
                result = reference(self.graph, self.model, self.q, xs, end, sealed_until=end, mode=self.options["mode"], trace=self.trace)
            elif self.algorithm == "python-stream-packed" or (streaming_phase and self.algorithm == "python-frontier"):
                self.last_execution["algorithm"] = "python-stream-packed"
                result = streaming(self.graph, self.model, self.q, xs, end, sealed_until=end, trace=self.trace,
                                    **{k: v for k, v in py_options.items() if k != "prefill"})
            elif self.algorithm == "python-frontier":
                result = frontier(self.graph, self.model, self.q, xs, end, sealed_until=end, trace=self.trace, **py_options)
            else:
                result = specialized_blocks.run(self.graph, self.model, self.q, xs, end, sealed_until=end,
                                                 topology=self.algorithm.removeprefix("python-"), **py_options)
        self.q, self.position = result.continuation, stop
        return result

    def detach(self):
        self.q = self.q.detach()
        if self.compiled:
            from types import SimpleNamespace
            graph = SimpleNamespace(identity=self.compiled.encoded_graph.identity)
            q = from_continuation(graph, self.eq).detach()
            self.eq = to_continuation(core, graph, self.compiled.encoded_graph, q)
        elif self.eg:
            self.eq = self.eq.detach()

    def cuts(self):
        pattern = self.config.get("execution_pattern", "whole")
        stop = self.config["sequence"]
        if pattern == "streaming":
            return [(t, "streaming") for t in range(1, stop+1)]
        if pattern == "prefill-stream":
            prefix = self.config.get("prefill_positions", stop//2)
            if not 0 < prefix < stop:
                raise ValueError("prefill-stream requires a nonempty prefix and decode suffix")
            return [(prefix, "prefill")]+[(t, "streaming") for t in range(prefix+1, stop+1)]
        if pattern != "whole":
            raise ValueError("unknown execution pattern")
        window = self.config.get("training_window") or stop
        return [(min(t, stop), "whole") for t in range(window, stop+window, window)]
