"""Round-three decoders, initialization identities and parameter activity.

Activity means unique parameter elements used in one external-token execution
slot. It excludes the tied vocabulary matrix for matching, includes the final
normalizer, and does not count padding or cached activations as new parameters.
Tide's prepare/reception/scoring work runs for every scheduled slot; its Full
parameters are counted from the *actual* finish controls, before row packing.
Thus shared bases are unioned across nodes/ticks, while private adapters remain
private. This is neither a gradient-support statistic nor a batch union.
"""
from __future__ import annotations

import copy
import math
import types
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .models import (AttentionState, CausalAttention, LMOutput, TideLM as _TideLM,
                     TransformerLM as _TransformerLM, TransformerState,
                     _check_tokens, detach_state)


INITIALIZATION_VERSIONS = {
    "tide": "upstream-default-lm-boundary-v1",
    "transformer": "dense-residual-depth-v1",
    "moe": "sparse-moe-residual-depth-v1",
}
ACTIVITY_VERSION = "unique-external-token-parameters-v1"
TIMED_LAST_NONEMPTY = "timed_last_nonempty_tick_in_period_v1"


def resolve_readout_policy(spec: dict, *, settle: bool) -> str:
    default = "settle_per_input_position" if settle else "timed_last_tick_before_next_external_input"
    policy = spec.get("readout_policy", default)
    allowed = {default} if settle else {default, TIMED_LAST_NONEMPTY}
    if policy not in allowed:
        raise ValueError(f"unsupported readout policy {policy!r} for this timing binding")
    return policy


def last_nonempty_readout(hidden: torch.Tensor, strengths: torch.Tensor, period: int):
    """Select the latest emission *within* each already consumed input period.

    No state is advanced and no flush/future external input is introduced.
    ``-1`` records an entirely empty period, whose emitted vector is zero.
    Selection is independent for each sample; all nodes at a tick retain the
    upstream weighted readout. The discrete support decision is not replaced
    by a soft approximation.
    """
    if hidden.ndim != 3 or strengths.shape != hidden.shape[:-1] or period < 1 or hidden.shape[1] % period:
        raise ValueError("readout requires aligned [batch, whole logical periods, width]")
    b, ticks, width = hidden.shape
    slots = hidden.reshape(b, ticks//period, period, width)
    mass = strengths.reshape(b, ticks//period, period)
    positions = torch.arange(period, device=hidden.device, dtype=torch.int64)
    selected = torch.where(mass > 0, positions, -torch.ones_like(positions)).amax(-1)
    values = slots.gather(2, selected.clamp_min(0)[..., None, None].expand(b, ticks//period, 1, width)).squeeze(2)
    values = torch.where((selected >= 0)[..., None], values, torch.zeros_like(values))
    return values, selected


@dataclass
class MoEOutput(LMOutput):
    auxiliary_loss: torch.Tensor | None = None


def parameter_counts(model: nn.Module) -> dict:
    """Count actual unique Parameter objects, including tied weights once."""
    params = {id(p): p for p in model.parameters()}
    embedding = getattr(model, "embedding", None)
    excluded = {id(p) for p in embedding.parameters()} if embedding is not None else set()
    whole = sum(p.numel() for p in params.values())
    vocabulary = sum(p.numel() for key, p in params.items() if key in excluded)
    return {"whole_model_parameters": whole, "vocabulary_parameters": vocabulary,
            "backbone_total_parameters": whole - vocabulary,
            "unique_parameter_tensors": len(params), "tied_vocabulary_counted_once": True}


def _activity(counts: dict, active: torch.Tensor, *, extra: dict | None = None) -> dict:
    active = active.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    total = counts["backbone_total_parameters"]
    if not active.numel() or bool((active < 0).any()) or bool((active > total).any()):
        return dict(qualified=False, version=ACTIVITY_VERSION,
                    reason="invalid per-token parameter coverage", **counts)
    mean = active.mean().item()
    return dict(qualified=True, version=ACTIVITY_VERSION, **counts,
                mean_active_backbone_parameters=mean,
                mean_active_whole_model_parameters=mean + counts["vocabulary_parameters"],
                active_ratio=mean / total, measured_external_tokens=active.numel(),
                min_active_backbone_parameters=int(active.min().item()),
                max_active_backbone_parameters=int(active.max().item()),
                attribution="unique parameter elements per external execution slot; Timed ticks unioned",
                vocabulary_policy="excluded from match; tied full output matrix included once in whole-model count",
                padding_policy="unpublished rectangular/power2 packing rows excluded",
                **(extra or {}))


def count_owner_activity(module: nn.Module, selected: torch.Tensor) -> torch.Tensor:
    """Count one NodeLinear/NodeVector's participating rows per token.

    ``selected[owner, token]`` is actual semantic participation. Sharing may
    reduce the base union, but never collapses private low-rank adapters.
    """
    from tide.parameters import NodeLinear, NodeVector
    if not isinstance(module, (NodeLinear, NodeVector)):
        raise TypeError("activity requires NodeLinear or NodeVector")
    if selected.ndim != 2 or selected.shape[0] != len(module._host_groups):
        raise ValueError("owner mask does not match the parameter ownership")
    selected = selected.detach().to(device="cpu", dtype=torch.bool)
    grouped_parameter = module.weight if isinstance(module, NodeLinear) else module.value
    groups = torch.tensor(module._host_groups, dtype=torch.int64)
    groups_used = torch.zeros((grouped_parameter.shape[0], selected.shape[1]), dtype=torch.int32)
    groups_used.index_add_(0, groups, selected.to(torch.int32))
    group_count = (groups_used > 0).sum(0)
    result = group_count * math.prod(grouped_parameter.shape[1:])
    if isinstance(module, NodeLinear):
        if module.bias is not None:
            result += group_count * math.prod(module.bias.shape[1:])
        if module.shared:
            result += selected.sum(0) * (math.prod(module.left.shape[1:]) + math.prod(module.right.shape[1:]))
    return result


def _traced_memory_finish(memory, context, control):
    meter = memory._round3_activity_meter
    if meter.capture is not None:
        meter._record(memory._round3_activity_kind, context[2], control)
    return type(memory).finish(memory, context, control)


class _TideActivity:
    """Coverage manifest for the pinned current Tide implementation.

    Explicit module coverage fails closed if upstream adds a parameter-bearing
    path. It works independently of fused projections, region batching,
    checkpoint recomputation and FFN padding because finish controls retain the
    original [owner, batch, logical-position] axes.
    """
    def __init__(self, model: "TideLM"):
        from tide.kv import KVMemory
        from tide.parameters import NodeLinear, NodeVector
        from tide.ssd_memory import SSDMemory
        self.model = model
        core = model.core
        self.dynamic: dict[str, list[nn.Module]] = {}
        self.capture: dict | None = None
        self.reasons: list[str] = []
        covered: set[int] = set()
        dynamic_ids: set[int] = set()

        def cover(module):
            covered.update(id(p) for p in module.parameters())

        # These are dense over all topology owners and logical slots, including
        # the documented zero-reception computation and regional contractions.
        for module in (core.inputs, model.output_norm, core.receive.a, core.receive.b,
                       core.receive.embedding, core.describe, core.regional):
            cover(module)
        if core.receive.profile != "source_residual":
            for name in ("q", "k", "v", "o", "sa_embedding"):
                cover(getattr(core.receive, name))
        for kind, memory in core.memories.items():
            if type(memory) is KVMemory:
                always, selected = ("kv", "summary"), ("query", "output", "ffn")
            elif type(memory) is SSDMemory:
                always = ("input", "summary", "read", "a_log", "dt_bias", "conv_kernel", "conv_bias")
                selected = ("skip", "rho", "output", "ffn")
            else:
                self.reasons.append(f"unsupported memory module {type(memory).__name__}")
                continue
            for name in always:
                cover(getattr(memory, name))
            leaves = []
            for name in selected:
                root = getattr(memory, name)
                for module in root.modules():
                    if isinstance(module, (NodeLinear, NodeVector)):
                        leaves.append(module)
                        for p in module.parameters(recurse=False):
                            if id(p) in dynamic_ids:
                                self.reasons.append("unexpected parameter alias between Full projections")
                            dynamic_ids.add(id(p))
                            covered.add(id(p))
                if {id(p) for p in root.parameters()} - dynamic_ids:
                    self.reasons.append(f"uncovered Full parameter in {kind}.{name}")
            self.dynamic[kind] = leaves
            # Instance-local wrapper, no global monkeypatch and no state_dict
            # mutation. Backward checkpoint recomputation sees capture=None.
            # A bound top-level function also preserves ownership under deepcopy.
            memory._round3_activity_meter = self
            memory._round3_activity_kind = kind
            memory.finish = types.MethodType(_traced_memory_finish, memory)
        params = {id(p): p for p in model.parameters() if id(p) != id(model.embedding.weight)}
        missing = params.keys() - covered
        if missing:
            names = [name for name, p in model.named_parameters() if id(p) in missing]
            self.reasons.append("uncovered parameters: " + ", ".join(names))
        self.always_count = sum(p.numel() for key, p in params.items() if key not in dynamic_ids)

    def begin(self, tokens):
        if self.capture is not None:
            raise RuntimeError("nested activity capture")
        self.capture = {"batch": tokens.shape[0], "length": tokens.shape[1], "owners": {},
                        "visited": {}, "errors": [], "full_rows": 0}

    def _record(self, kind, owners, control):
        from tide.parameters import owner_ids
        current = self.capture
        b, length = current["batch"], current["length"]
        period = 1 if self.model.core.settle else self.model.core.graph.period
        if control.ndim != 3 or control.shape[1:] != (b, length * period):
            current["errors"].append("finish controls no longer retain original batch/time axes")
            return
        ids = owner_ids(owners).detach().to(device="cpu", dtype=torch.int64)
        mask = (control.detach() > 0).to("cpu")
        current["full_rows"] += int(mask.sum())
        mask = mask.reshape(len(ids), b, length, period).any(-1).reshape(len(ids), -1)
        count = len(self.model.core.memory_ids[kind])
        bank = current["owners"].setdefault(kind, torch.zeros((count, b * length), dtype=torch.bool))
        visited = current["visited"].setdefault(kind, torch.zeros(count, dtype=torch.bool))
        if ids.numel() != mask.shape[0] or ids.unique().numel() != ids.numel():
            current["errors"].append("invalid or repeated memory owners")
            return
        bank[ids] |= mask
        visited[ids] = True

    def end(self):
        current, self.capture = self.capture, None
        reasons = list(self.reasons) + current["errors"]
        for kind in self.dynamic:
            if kind not in current["visited"] or not bool(current["visited"][kind].all()):
                reasons.append(f"incomplete finish coverage for memory {kind}")
        counts = parameter_counts(self.model)
        if reasons:
            return dict(qualified=False, version=ACTIVITY_VERSION, **counts, reasons=reasons)
        active = torch.full((current["batch"] * current["length"],), self.always_count, dtype=torch.int64)
        for kind, leaves in self.dynamic.items():
            for module in leaves:
                active += count_owner_activity(module, current["owners"][kind])
        return _activity(counts, active, extra={"method": "actual-finish-controls-with-explicit-parameter-coverage",
                         "always_active_backbone_parameters": self.always_count,
                         "semantic_full_owner_tick_rows": current["full_rows"]})


class TideLM(_TideLM):
    def __init__(self, spec, vocab_size):
        super().__init__(spec, vocab_size)
        self.readout_policy = resolve_readout_policy(spec, settle=self.core.settle)
        self._activity_meter = _TideActivity(self)

    def forward(self, tokens, state=None, record_work=False):
        _check_tokens(tokens)
        if record_work:
            self._activity_meter.begin(tokens)
        try:
            result = self.core(self.embedding(tokens), state, record_work=record_work)
            hidden = result.readout()
            strengths = result.strengths.sum(0)
            selected = None
            if not self.core.settle:
                b, length = tokens.shape
                if self.readout_policy == TIMED_LAST_NONEMPTY:
                    hidden, selected = last_nonempty_readout(hidden, strengths, self.core.graph.period)
                else:
                    hidden = hidden.reshape(b, length, self.core.graph.period, self.d)[:, :, -1]
                    observed = strengths.reshape(b, length, self.core.graph.period)[:, :, -1]
                    selected = torch.where(observed > 0, self.core.graph.period-1, -1)
            logits = F.linear(self.output_norm(hidden), self.embedding.weight)
            work = None
            if record_work:
                empty = (strengths <= 0) if selected is None else (selected < 0)
                work = dict(result.work or {})
                work.update(family="tide", readout_policy=self.readout_policy,
                            input_positions=tokens.numel(), output_positions=tokens.numel(),
                            empty_output_positions=int(empty.sum().item()),
                            empty_logical_output_slots=int((strengths <= 0).sum().item()),
                            logical_ticks=int(result.times.numel()), period=int(self.core.graph.period),
                            activity=self._activity_meter.end(), health={"logits": tensor_health(logits)})
                if selected is not None:
                    chosen = selected.detach().to("cpu")
                    start = 0 if state is None else state.position
                    external = torch.arange(start, start+tokens.shape[1], dtype=torch.int64)
                    times = torch.where(chosen >= 0, external*self.core.graph.period+chosen+1, -1)
                    work.update(selected_tick_in_period=chosen.tolist(), selected_logical_time=times.tolist(),
                                readout_empty_sentinel=-1, readout_inserts_flush=False)
            return LMOutput(logits, result.state, work)
        finally:
            self._activity_meter.capture = None


class TransformerLM(_TransformerLM):
    def forward(self, tokens, state=None, record_work=False):
        result = super().forward(tokens, state, record_work)
        if record_work:
            counts = parameter_counts(self)
            result.work["activity"] = _activity(counts, torch.full((tokens.numel(),),
                                                       counts["backbone_total_parameters"]),
                                                extra={"method": "dense-all-parameters-per-token"})
            result.work["health"] = {"logits": tensor_health(result.logits)}
        return result


class Expert(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.gate_up = nn.Linear(d, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, -1)
        return self.down(F.silu(gate) * up)


class SparseMoE(nn.Module):
    """Top-k token dispatch without capacity drops or unselected expert work."""
    def __init__(self, d, hidden, experts, top_k):
        super().__init__()
        if not 1 <= top_k <= experts:
            raise ValueError("MoE requires 1 <= top_k <= experts")
        self.router = nn.Linear(d, experts, bias=False)
        self.experts = nn.ModuleList(Expert(d, hidden) for _ in range(experts))
        self.top_k = top_k

    def forward(self, x, record_work=False):
        flat = x.reshape(-1, x.shape[-1])
        with torch.autocast(x.device.type, enabled=False):
            logits = F.linear(flat.to(self.router.weight.dtype), self.router.weight)
            probability = logits.softmax(-1)
            chosen_logits, chosen = logits.topk(self.top_k, dim=-1)
            weights = chosen_logits.softmax(-1)
        # index_add preserves router and expert gradients. No expert sees rows
        # routed elsewhere, even if another token in this batch selected it.
        result = torch.zeros_like(flat)
        dispatched = []
        for index, expert in enumerate(self.experts):
            rows, slots = (chosen == index).nonzero(as_tuple=True)
            dispatched.append(rows.numel())
            if rows.numel():
                computed = expert(flat.index_select(0, rows))
                contribution = computed * weights[rows, slots, None].to(computed.dtype)
                result = result.index_add(0, rows, contribution.to(result.dtype))
        fraction = F.one_hot(chosen, len(self.experts)).float().mean((0, 1))
        auxiliary = len(self.experts) * (fraction.detach() * probability.mean(0)).sum()
        work = None
        if record_work:
            # The route union is local to each token, never the entire batch.
            route = chosen.detach().to("cpu")
            unique_count = torch.stack([(route == expert).any(-1) for expert in range(len(self.experts))]).sum(0)
            expert_parameters = sum(p.numel() for p in self.experts[0].parameters())
            work = {"selected_expert_parameters_per_token": unique_count * expert_parameters,
                    "dispatch_rows_by_expert": dispatched, "top_k": self.top_k,
                    "experts": len(self.experts), "capacity_drops": 0}
        return result.reshape_as(x), auxiliary, work


class MoEBlock(nn.Module):
    def __init__(self, d, heads, hidden, window, experts, top_k):
        super().__init__()
        self.attention_norm = nn.RMSNorm(d, eps=1e-5)
        self.attention = CausalAttention(d, heads, window)
        self.ffn_norm = nn.RMSNorm(d, eps=1e-5)
        self.moe = SparseMoE(d, hidden, experts, top_k)

    def forward(self, x, state, start, record_work):
        attention, following = self.attention(self.attention_norm(x), state, start)
        x = x + attention
        update, auxiliary, work = self.moe(self.ffn_norm(x), record_work)
        return x + update, following, auxiliary, work


class MoELM(nn.Module):
    def __init__(self, spec, vocab_size):
        super().__init__()
        self.model_spec = copy.deepcopy(spec)
        self.family, self.vocab_size = "moe", vocab_size
        self.d = int(spec.get("d", 128))
        layers, heads = int(spec.get("layers", 4)), int(spec.get("heads", 4))
        hidden = int(spec.get("ffn_width", 4 * self.d))
        experts, top_k = int(spec.get("experts", 8)), int(spec.get("top_k", 2))
        self.window = int(spec.get("window", 128))
        if min(self.d, layers, heads, hidden, experts, top_k, self.window) < 1:
            raise ValueError("MoE dimensions must be positive")
        self.embedding = nn.Embedding(vocab_size, self.d)
        self.blocks = nn.ModuleList(MoEBlock(self.d, heads, hidden, self.window, experts, top_k) for _ in range(layers))
        self.output_norm = nn.RMSNorm(self.d, eps=1e-5)
        self.readout_policy = "causal_per_input_position"
        self.apply(_TransformerLM._initialize)
        with torch.no_grad():
            for block in self.blocks:
                block.attention.out.weight.div_(math.sqrt(2 * layers))
                for expert in block.moe.experts:
                    expert.down.weight.div_(math.sqrt(2 * layers))

    def forward(self, tokens, state=None, record_work=False):
        _check_tokens(tokens)
        if state is not None and not isinstance(state, TransformerState):
            raise TypeError("expected TransformerState")
        if state is not None and len(state.layers) != len(self.blocks):
            raise ValueError("MoE cache has a different number of layers")
        start = 0 if state is None else state.position
        if type(start) is not int or start < 0:
            raise ValueError("state position must be a nonnegative integer")
        hidden, following, auxiliary, routed = self.embedding(tokens), [], [], []
        for index, block in enumerate(self.blocks):
            hidden, cache, loss, measured = block(hidden, None if state is None else state.layers[index], start, record_work)
            following.append(cache)
            auxiliary.append(loss)
            routed.append(measured)
        logits = F.linear(self.output_norm(hidden), self.embedding.weight)
        work = None
        if record_work:
            counts = parameter_counts(self)
            all_expert_params = sum(p.numel() for block in self.blocks for p in block.moe.experts.parameters())
            active = torch.full((tokens.numel(),), counts["backbone_total_parameters"] - all_expert_params)
            for item in routed:
                active += item.pop("selected_expert_parameters_per_token")
            work = dict(family="moe", input_positions=tokens.numel(), output_positions=tokens.numel(),
                        empty_output_positions=0, readout_policy=self.readout_policy,
                        routing=routed, activity=_activity(counts, active, extra={
                            "method": "actual-unique-routes-plus-shared-attention-router-and-norms"}),
                        health={"logits": tensor_health(logits)})
        return MoEOutput(logits, TransformerState(start + tokens.shape[1], tuple(following)),
                         work, torch.stack(auxiliary).mean())


def tensor_health(value: torch.Tensor) -> dict:
    value = value.detach().float()
    finite = torch.isfinite(value)
    valid = torch.where(finite, value, torch.zeros_like(value))
    count = value.numel()
    maximum = valid.abs().max() if count else valid.new_zeros(())
    scale = torch.where(maximum > 0, maximum, torch.ones_like(maximum))
    rms = (valid/scale).square().mean().sqrt()*maximum if count else maximum
    return {"elements": count, "nonfinite": int((~finite).sum().item()),
            "zeros": int((value == 0).sum().item()),
            "rms": float(rms.item()), "max_abs": float(maximum.item())}


def parameter_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    """Capture before a real optimizer update, only at planned health samples."""
    return {name: p.detach().float().cpu().clone() for name, p in model.named_parameters()}


def health_snapshot(model: nn.Module, before: dict[str, torch.Tensor] | None = None) -> dict:
    """Raw per-tensor evidence; sparse/zero gradients are not a failure verdict."""
    entries = {}
    for name, parameter in model.named_parameters():
        row = {"parameter": tensor_health(parameter),
               "gradient": None if parameter.grad is None else tensor_health(parameter.grad)}
        if before is not None:
            if name not in before or before[name].shape != parameter.shape:
                raise ValueError("update reference belongs to a different model")
            delta = parameter.detach().float().cpu() - before[name]
            row["update"] = tensor_health(delta)
            row["update_to_parameter_rms"] = row["update"]["rms"] / max(row["parameter"]["rms"], 1e-30)
        entries[name] = row
    return {"version": "parameter-gradient-update-health-v1", "parameters": entries,
            "zero_gradient_policy": "observation only; contextual qualification required"}


def build_model(spec: dict, vocab_size: int, init_recipe: str = "default") -> nn.Module:
    if type(vocab_size) is not int or vocab_size < 2:
        raise ValueError("vocab_size must be an integer >= 2")
    family = spec.get("family")
    if family not in INITIALIZATION_VERSIONS:
        raise ValueError(f"unsupported model family: {family!r}")
    version = INITIALIZATION_VERSIONS[family]
    # No alternative is invented before a measured need. A versioned default
    # is the finite initial menu, and is part of the checkpoint identity.
    if init_recipe not in ("default", version):
        raise ValueError(f"unsupported initialization {init_recipe!r} for {family}")
    model = {"tide": TideLM, "transformer": TransformerLM, "moe": MoELM}[family](spec, vocab_size)
    model.initialization_recipe = version
    return model
