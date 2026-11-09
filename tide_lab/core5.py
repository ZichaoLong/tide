"""Small complete TIDE models using the installed public core-5 API.

This module owns embeddings/readout only. Scheduling, local functions,
autograd and shared graph parameters remain in the pinned tidegraph package.
"""
import copy
import torch
from tidegraph import Continuation, ExecutionPlacement
from tidegraph.planning.recipes import prepare
from tidegraph.planning.materialize import materialize
from tidegraph.settle import SettleGraph
from tidegraph.weighted import layered_inputs, readout
from tidegraph.weighted.execution import eager_runner


class Core5Encoder(torch.nn.Module):
    """[batch, positions, width] -> final-position weighted port readout.

    All examples start with empty private state; the full episode participates
    in BPTT. No continuation survives an optimizer update. The packed public
    executor batches actual samples/positions; observed operator batch sizes
    are reported instead of inferred from the logical batch label.
    """
    def __init__(self, width=16, topology="fc", depth=2, region_width=4,
                 send="dense", seed=7, temperature=1.0, dtype=torch.float32,
                 memory_read="qry", branches=2, offsets=(0, 1), family="kv",
                 history=512, support_policy="strict", recipe=None,
                 max_materialize_bytes=64 * 1024 ** 2, eager_implementation="python",
                 placement_preset=None, native_library=None):
        super().__init__()
        if send not in {"dense", "sparse"}:
            raise ValueError("send must be dense (Shannon) or sparse (quadratic)")
        if region_width < 4 or region_width % 2:
            raise ValueError("first-batch regional comparison requires even width >=4 and k>=2")
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.width, self.send = width, send
        self.recipe = dict(topology=topology, family=family, width=width,
                           depth=depth, region_width=region_width, branches=branches,
                           offsets=list(offsets), profile="baseline", source_mode="region",
                           sparse_denominator=2, selection="regional_total",
                           regularizer="ent" if send == "dense" else "quad",
                           ent_alpha=1.0 if send == "dense" else None, history=history,
                           memory_overrides=dict(memory_read=memory_read),
                           region_overrides=dict(allocation_temperature=float(temperature)))
        if support_policy != "strict":
            self.recipe["region_overrides"]["support_policy"] = support_policy
        if recipe is not None:
            self.recipe = copy.deepcopy(recipe)
            if self.recipe.get("width") != width:
                raise ValueError("explicit recipe width must match task embedding width")
            self.send = "explicit-recipe"
        if eager_implementation not in {"python", "native"}:
            raise ValueError("eager implementation must be python/native")
        if type(max_materialize_bytes) is not int or max_materialize_bytes <= 0:
            raise ValueError("materialization requires a positive explicit byte budget")
        self.eager_implementation = eager_implementation
        self.placement_preset, self.native_library = placement_preset, native_library
        self.plan, self.inventory = prepare(self.recipe, samples=1, positions=1, seed=seed)
        self.binding, _, _ = materialize(self.plan, max_bytes=max_materialize_bytes, dtype=dtype)
        self.graph_model = self.binding.model
        self.spec = SettleGraph(self.binding.graph, tuple(self.plan["graph"]["ranks"]),
                                binding="ports", time_stride=self.plan["stride"])
        self.last_metrics = {}
        self.observe = False

    @property
    def model_id(self):
        return self.plan["model_id"]

    def facts(self):
        return dict(model_id=self.model_id, recipe=self.recipe,
                    nodes=len(self.binding.graph.nodes), regions=len(self.binding.graph.regions),
                    edges=len(self.binding.graph.edges),
                    graph_parameters=sum(p.numel() for p in self.graph_model.parameters() if p.requires_grad),
                    state_reset="fresh sequence", bptt="full episode",
                    execution=f"public eager {self.eager_implementation} packed; one host scheduler worker",
                    placement_preset=self.placement_preset)

    def run_features(self, features, *, implementation="eager", chunk_size=None, trace=True):
        """Public-result access for independent audits, with no oracle-driven route."""
        if features.ndim != 3 or features.shape[-1] != self.width or features.shape[1] < 1:
            raise ValueError("features must be nonempty [batch,positions,width]")
        if implementation == "reference":
            from tidegraph.reference import run
            execute = lambda *a, **kw: run(self.binding.graph, self.graph_model, *a, **kw)
        elif implementation in {"eager", "streaming"}:
            payload = next(self.graph_model.parameters()).device
            placement = ExecutionPlacement(preset=self.placement_preset or ("cpu" if payload.type == "cpu" else "mixed-c"),
                                           scoring_dtype="float64" if features.dtype == torch.float64 else "float32")
            execute = eager_runner(self.binding, self.eager_implementation, placement,
                                   native_library=self.native_library,
                                   streaming=implementation == "streaming", workers=1,
                                   packing="scalar" if implementation == "streaming" else "packed")
        else:
            raise ValueError("unknown implementation")
        batch, length, _ = features.shape
        q = Continuation(self.binding.graph.identity, batch)
        chunk_size = length if chunk_size is None else chunk_size
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        all_outputs, all_trace, all_messages, stats = [], [], [], {}
        result = None
        for start in range(0, length, chunk_size):
            end = min(length, start + chunk_size)
            inputs = layered_inputs(self.spec, features[:, start:end], start_position=start)
            options = dict(sealed_until=end * self.spec.stride)
            # The public native factory returns Native.run, whose trace policy
            # is constructor-owned (complete records); Python takes it per call.
            if implementation == "reference" or self.eager_implementation == "python":
                options['trace'] = trace
            result = execute(q, inputs, end * self.spec.stride, **options)
            q = result.continuation
            all_outputs.extend(result.outputs)
            all_trace.extend(result.trace)
            all_messages.extend(result.messages)
            for name, value in result.stats.items():
                stats[name] = max(stats.get(name, 0), value) if name.startswith("max_") else stats.get(name, 0) + value
        result.outputs, result.trace, result.messages, result.stats = all_outputs, all_trace, all_messages, stats
        return result

    def project(self, result, features, position=-1):
        position = features.shape[1] - 1 if position == -1 else position
        coordinates = [(self.spec.stride * position + self.spec.rank(node), port)
                       for port, node in enumerate(self.binding.graph.outputs)]
        return torch.stack([readout(result.outputs, coordinates, features[b, position], batch=b)
                            for b in range(features.shape[0])])

    def forward(self, features):
        result = self.run_features(features, trace=self.observe)
        stats = result.stats
        candidates, full = stats.get("candidate_events", 0), stats.get("selected_events", 0)
        self.last_metrics = {"work/candidates": candidates, "work/full": full,
                             "work/state_updates": candidates,
                             "work/read_events": candidates,
                             "work/messages": stats.get("visited_edges", 0),
                             "work/source_rows": stats.get("source_rows", 0),
                             "activity/full_fraction": full / max(1, candidates),
                             "work/max_full_batch": stats.get("max_full_batch", 0),
                             "work/max_state_batch": stats.get("max_state_batch", 0),
                             "work/max_state_sequence": stats.get("max_state_sequence", 0),
                             "work/max_region_batch": stats.get("max_region_batch", 0),
                             "work/region_blocks": stats.get("region_blocks", 0)}
        if self.observe and result.trace:
            controls = torch.stack([e["control"] for e in result.trace]).detach()
            self.last_metrics["activity/send_strength"] = float(controls.sum().cpu())
            self.last_metrics["activity/nonpositive_fraction"] = float((controls <= 0).float().mean().cpu())
            for region in range(len(self.binding.graph.regions)):
                events = [e for e in result.trace if self.binding.graph.nodes[e["node"]].region == region]
                self.last_metrics[f"region/{region}/candidates"] = len(events)
                self.last_metrics[f"region/{region}/full"] = sum(e["active"] for e in events)
        return self.project(result, features)


class Core5Classifier(torch.nn.Module):
    def __init__(self, vocab_size, value_vocab_size, width=16, seed=7, **encoder_options):
        super().__init__()
        self.encoder = Core5Encoder(width=width, seed=seed, **encoder_options)
        dtype = next(self.encoder.parameters()).dtype
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.embedding = torch.nn.Embedding(vocab_size, width, dtype=dtype)
            self.head = torch.nn.Linear(width, value_vocab_size, dtype=dtype)
        self.last_metrics = {}

    def forward(self, token_ids):
        features = self.encoder(self.embedding(token_ids))
        self.last_metrics = dict(self.encoder.last_metrics)
        return self.head(features)

    def facts(self):
        return dict(self.encoder.facts(),
                    total_parameters=sum(p.numel() for p in self.parameters()),
                    trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad))
