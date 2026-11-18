"""Untimed cache clock/support checks against observed positive receptions.

The independent ledger retains only original int64 write times inside each
owner's window. It never copies KV payloads or keeps the whole input history.
Ordinary prepare hooks run in actual region workers; shared ledgers are locked.
Normal inference timing and production state transitions are unchanged.
"""
from collections import deque
import hashlib
import json
from threading import Lock

import torch

from benchmarks.inference import OutputDiagnostic
from tide.parameters import owner_ids


def _digest(times):
    return hashlib.sha256(json.dumps(times, separators=(",", ":")).encode()).hexdigest()


def inspect_kv_bank(bank, *, owners, batch, capacity, window, processed_time, expected):
    """Check one actual bank against independently observed per-owner writes."""
    failures, rows = [], []
    shape = (len(owners), batch)
    if bank.times.ndim != 3 or tuple(bank.times.shape[:2]) != shape:
        return dict(rows=[], failures=["times_shape"], capacity=capacity, window=window)
    slots = bank.times.shape[-1]
    if bank.times.dtype != torch.int64:
        failures.append("times_not_int64")
    if slots > capacity:
        failures.append("capacity_exceeded")
    if bank.weight.shape != bank.times.shape:
        failures.append("weight_shape")
    for name in ("keys", "values", "features"):
        value = getattr(bank, name)
        if value.ndim != 4 or value.shape[:3] != bank.times.shape:
            failures.append(name+"_shape")
    if failures:
        return dict(rows=[], failures=failures, capacity=capacity, observed_slots=slots, window=window)
    times, weight = bank.times.detach().cpu(), bank.weight.detach().cpu()
    if not bool(torch.isfinite(weight).all()) or bool((weight < 0).any()):
        failures.append("invalid_weight")
    for local, owner in enumerate(owners):
        for sample in range(batch):
            actual = times[local, sample][weight[local, sample] > 0].tolist()
            target = tuple(expected.get((owner, sample), ()))
            last_write = target[-1] if target else 0
            errors = []
            if any(t < 1 or t > processed_time for t in actual):
                errors.append("timestamp_outside_processed_clock")
            if any(a >= b for a, b in zip(actual, actual[1:])):
                errors.append("positive_times_not_strictly_increasing")
            # Python int differences avoid overflow and never round 2**53+.
            ages = [last_write-t for t in actual]
            if any(age < 0 or age >= window for age in ages):
                errors.append("outside_last_write_window")
            if tuple(actual) != target:
                errors.append("retained_times_differ_from_positive_writes")
            rows.append(dict(owner=owner, sample=sample, configured_capacity=capacity, observed_slots=slots,
                positive_entries=len(actual), expected_positive_entries=len(target), empty=not actual,
                full=len(actual) == capacity, first_positive_time=min(actual) if actual else None,
                last_positive_time=max(actual) if actual else None, last_write_time=last_write,
                processed_through_time=processed_time, maximum_age_from_last_write=max(ages) if ages else None,
                retained_times_sha256=_digest(actual), expected_times_sha256=_digest(target), failures=errors))
    return dict(capacity=capacity, observed_slots=slots, window=window, rows=rows, failures=failures,
                devices=dict(times=str(bank.times.device), keys=str(bank.keys.device)),
                dtypes=dict(times=str(bank.times.dtype), weights=str(bank.weight.dtype), keys=str(bank.keys.dtype)))


def inspect_ssd_bank(bank, *, owners, batch, stride, processed_time, expected, last_writes):
    failures, rows = [], []
    shape, capacity, horizon = (len(owners), batch), 3*stride+1, 3*stride
    if tuple(bank.times.shape) != shape or bank.times.dtype != torch.int64:
        failures.append("raw_write_time_shape_or_dtype")
    if (bank.conv.times.ndim != 3 or tuple(bank.conv.times.shape[:2]) != shape or
            bank.conv.times.dtype != torch.int64 or bank.conv.values.ndim != 4 or
            bank.conv.values.shape[:3] != bank.conv.times.shape):
        failures.append("convolution_shape_or_time_dtype")
    if failures:
        return dict(rows=[], failures=failures, convolution_capacity=capacity)
    slots = bank.conv.times.shape[-1]
    if slots > capacity:
        failures.append("convolution_capacity_exceeded")
    raw, conv = bank.times.detach().cpu(), bank.conv.times.detach().cpu()
    for local, owner in enumerate(owners):
        for sample in range(batch):
            last = int(raw[local, sample])
            times = conv[local, sample].tolist()
            actual = [t for t in times if t >= 0]
            target = tuple(expected.get((owner, sample), ()))
            expected_last = last_writes.get((owner, sample), 0)
            errors = []
            if last != expected_last or last < 0 or last > processed_time:
                errors.append("raw_time_differs_from_last_positive_reception")
            if any(t < -1 for t in times) or times != sorted(times):
                errors.append("convolution_padding_or_time_order")
            if any(a >= b for a, b in zip(actual, actual[1:])):
                errors.append("convolution_times_not_strictly_increasing")
            if any(t < 1 or t > processed_time or not 0 <= expected_last-t <= horizon for t in actual):
                errors.append("convolution_time_outside_last_write_horizon")
            if tuple(actual) != target:
                errors.append("convolution_times_differ_from_positive_writes")
            rows.append(dict(owner=owner, sample=sample, raw_last_write_time=last,
                expected_last_write_time=expected_last, convolution_slots=slots, positive_entries=len(actual),
                first_positive_time=actual[0] if actual else None, last_positive_time=actual[-1] if actual else None,
                convolution_horizon=horizon, retained_times_sha256=_digest(actual),
                expected_times_sha256=_digest(target), failures=errors))
    return dict(rows=rows, failures=failures, convolution_capacity=capacity, observed_slots=slots,
                times_dtype=str(bank.times.dtype), convolution_times_dtype=str(bank.conv.times.dtype))


class WriteClockLedger:
    """Bounded timestamp oracle from actual prepare inputs, not returned caches."""
    def __init__(self, model, batch):
        self.model, self.batch = model, batch
        self.nodes = {kind: {index: node for node, index in ids.items()} for kind, ids in model.memory_ids.items()}
        self.expected, self.last, self.write_counts = {}, {}, {}
        self.failures, self.failure_count = [], 0
        self.maximum_retained_timestamps = self.retained_timestamps = 0
        self._lock, self._originals = Lock(), []

    def _fail(self, message):
        self.failure_count += 1
        if len(self.failures) < 32:
            self.failures.append(message)

    def observe(self, kind, mass, times, owners):
        if times.dtype != torch.int64 or times.shape != mass.shape or mass.ndim != 3 or mass.shape[1] != self.batch:
            with self._lock:
                self._fail(dict(kind=kind, reason="prepare_time_shape_or_dtype"))
            return
        ids = owner_ids(owners).detach().cpu().tolist()
        values = times.detach().cpu()
        support = (mass.detach() > 0).cpu()
        memory = self.model.memories[kind]
        span = memory.window-1 if kind == "K" else 3*memory.stride
        capacity = memory.slots if kind == "K" else span+1
        with self._lock:
            for local, parameter_owner in enumerate(ids):
                node = self.nodes[kind][parameter_owner]
                for sample in range(self.batch):
                    key = kind, node, sample
                    history = self.expected.setdefault(key, deque())
                    previous_size = len(history)
                    writes = values[local, sample][support[local, sample]].tolist()
                    self.write_counts[key] = self.write_counts.get(key, 0)+len(writes)
                    last = self.last.get(key, 0)
                    if writes and (writes[0] <= last or any(a >= b for a, b in zip(writes, writes[1:]))):
                        self._fail(dict(kind=kind, owner=node, sample=sample, reason="prepare_writes_not_strictly_after_previous_write"))
                    for value in writes:
                        history.append(value)
                    if writes:
                        last = writes[-1]
                    self.last[key] = last
                    while history and last-history[0] > span:
                        history.popleft()
                    if len(history) > capacity:
                        self._fail(dict(kind=kind, owner=node, sample=sample, reason="observed_writes_exceed_declared_capacity"))
                        # Bound a malformed stream too; never present it as a pass.
                        while len(history) > capacity:
                            history.popleft()
                    self.retained_timestamps += len(history)-previous_size
                    self.maximum_retained_timestamps = max(self.maximum_retained_timestamps, self.retained_timestamps)

    def __enter__(self):
        for kind, memory in self.model.memories.items():
            original, owned = memory.prepare, "prepare" in vars(memory)
            self._originals.append((memory, original, owned))
            def prepare(u, mass, times, owners, state=None, *, _original=original, _kind=kind):
                result = _original(u, mass, times, owners, state)
                self.observe(_kind, mass, times, owners)
                return result
            memory.prepare = prepare
        return self

    def __exit__(self, *exc):
        for memory, original, owned in self._originals:
            if owned:
                memory.prepare = original
            else:
                delattr(memory, "prepare")


def inspect_pending(model, state, batch, *, drained):
    failures, rows = [], []
    if model.settle:
        return dict(banks=[], failures=[] if not state.pending else ["Settle_has_pending"])
    if len(state.pending) != len(model.layer_nodes):
        return dict(banks=[], failures=["pending_layer_count"])
    for layer, bank in enumerate(state.pending):
        if hasattr(bank, "regions"):
            if len(bank.regions) != len(model.layer_regions[layer]):
                failures.append("pending_region_count"); continue
            chunks = [(model.graph.regions[region], chunk) for region, chunk in zip(model.layer_regions[layer], bank.regions)]
        else:
            chunks = [(model.layer_nodes[layer], bank)]
        for nodes, chunk in chunks:
            errors = []
            if tuple(chunk.values.shape) != (len(nodes), batch, 1, model.config.d) or tuple(chunk.strengths.shape) != (len(nodes), batch, 1):
                errors.append("pending_shape")
                outgoing = None
            else:
                strengths = chunk.strengths.detach().cpu()
                if not bool(torch.isfinite(chunk.values).all()):
                    errors.append("pending_nonfinite_values")
                if not bool(torch.isfinite(strengths).all()) or bool((strengths < 0).any()):
                    errors.append("pending_invalid_strength")
                outgoing = sum(int((strengths[i] > 0).sum()) for i, node in enumerate(nodes) if model.graph.outgoing[node])
                if drained and outgoing:
                    errors.append("drain_left_pending_nonterminal_sender")
            rows.append(dict(layer=layer, nodes=list(nodes), inferred_tick=state.position*model.graph.period,
                             outgoing_positive_pending=outgoing, failures=errors))
    return dict(banks=rows, failures=failures,
                drain_rule="only nodes with outgoing edges must have no pending senders; terminal output tails may remain at period=1")


class CacheMetadataDiagnostic:
    def __init__(self, model, schedule, batch, *, sample_every=64, on_snapshot=None):
        if type(sample_every) is not int or sample_every < 1:
            raise ValueError("sample_every must be a positive input-position interval")
        self.model, self.schedule, self.batch = model, schedule, batch
        self.sample_every, self.on_snapshot = sample_every, on_snapshot
        self.output = OutputDiagnostic()
        self.ledger = WriteClockLedger(model, batch)
        self.snapshots, self.previous_kv, self.evictions = [], {}, []
        self.ever_full, self.owner_keys, self.evicted_owners = set(), set(), set()
        self.calls, self.last_position, self.last_phase = 0, 0, None
        self.previous_state_position = 0
        self.output_time_failures, self.output_time_failure_count = [], 0

    def __call__(self, value, raw, phase, state):
        self.output(value, raw, phase, state)
        if self.model.settle:
            expected_times = torch.arange(self.previous_state_position, state.position, dtype=torch.int64)*self.model.graph.period+len(self.model.layer_nodes)
        else:
            expected_times = torch.arange(self.previous_state_position*self.model.graph.period+1,
                                          state.position*self.model.graph.period+1, dtype=torch.int64)
        if raw.times.dtype != torch.int64 or not torch.equal(raw.times.detach().cpu(), expected_times):
            self.output_time_failure_count += 1
            if len(self.output_time_failures) < 32:
                self.output_time_failures.append(dict(call=self.calls, phase=phase, reason="original_output_time_axis_mismatch"))
        self.previous_state_position = state.position
        sample = (self.calls == 0 or phase != self.last_phase or phase == "drain" or
                  state.position in (self.schedule.prefill, self.schedule.positions) or
                  state.position-self.last_position >= self.sample_every)
        self.calls += 1; self.last_phase = phase
        if not sample:
            return
        self.last_position = state.position
        row = dict(call=self.calls-1, phase=phase, state_position=state.position, kv=[], ssd=[], evictions=[])
        if len(state.memories) != len(self.model.graph.regions):
            row["failures"] = ["memory_region_count"]
        else:
            for region, memories in enumerate(state.memories):
                members = self.model.graph.regions[region]
                if set(memories) != {self.model.kinds[node] for node in members}:
                    row.setdefault("failures", []).append(f"region_{region}:memory_kind_mismatch")
                layer = self.model.graph.nodes[members[0]].layer
                processed = ((state.position-1)*self.model.graph.period+layer+1 if self.model.settle else
                             state.position*self.model.graph.period)
                for kind, bank in memories.items():
                    nodes = tuple(node for node in members if self.model.kinds[node] == kind)
                    expected = {(node, sample): tuple(self.ledger.expected.get((kind, node, sample), ()))
                                for node in nodes for sample in range(self.batch)}
                    memory = self.model.memories[kind]
                    if kind == "K":
                        inspected = inspect_kv_bank(bank, owners=nodes, batch=self.batch, capacity=memory.slots,
                                                   window=memory.window, processed_time=processed, expected=expected)
                        for owner in inspected["rows"]:
                            key = owner["owner"], owner["sample"]
                            self.owner_keys.add(key)
                            if owner["full"]:
                                self.ever_full.add(key)
                            before = self.previous_kv.get(key)
                            if (before is not None and before["first_positive_time"] is not None and
                                    owner["first_positive_time"] is not None and
                                    owner["first_positive_time"] > before["first_positive_time"]):
                                witness = dict(owner=key[0], sample=key[1], previous_snapshot=before["snapshot"],
                                    current_snapshot=len(self.snapshots), previous_first_time=before["first_positive_time"],
                                    current_first_time=owner["first_positive_time"], current_last_write_time=owner["last_write_time"],
                                    old_first_age_now=owner["last_write_time"]-before["first_positive_time"], window=memory.window,
                                    retained_write_set_verified=(not inspected["failures"] and not owner["failures"] and
                                        before["bank_valid"] and not before["failures"] and not self.ledger.failure_count and
                                        owner["last_write_time"]-before["first_positive_time"] >= memory.window))
                                row["evictions"].append(witness)
                                self.evictions.append(witness)
                                if witness["retained_write_set_verified"]:
                                    self.evicted_owners.add(key)
                            self.previous_kv[key] = dict(owner, snapshot=len(self.snapshots), bank_valid=not inspected["failures"])
                        inspected["region"] = region
                        row["kv"].append(inspected)
                    elif kind == "S":
                        last = {(node, sample): self.ledger.last.get((kind, node, sample), 0)
                                for node in nodes for sample in range(self.batch)}
                        inspected = inspect_ssd_bank(bank, owners=nodes, batch=self.batch, stride=memory.stride,
                                                    processed_time=processed, expected=expected, last_writes=last)
                        inspected["region"] = region
                        row["ssd"].append(inspected)
        row["pending"] = inspect_pending(self.model, state, self.batch, drained=phase == "drain")
        self.snapshots.append(row)
        if self.on_snapshot is not None:
            self.on_snapshot(self)

    def report(self, state, *, require_kv_eviction=False):
        failed = []
        for snapshot in self.snapshots:
            for reason in snapshot.get("failures", ()):
                failed.append(dict(call=snapshot["call"], reason=reason))
            for kind in ("kv", "ssd"):
                for bank in snapshot[kind]:
                    for reason in bank["failures"]:
                        failed.append(dict(call=snapshot["call"], kind=kind, region=bank["region"], reason=reason))
                    for row in bank["rows"]:
                        if row["failures"]:
                            failed.append(dict(call=snapshot["call"], kind=kind, region=bank["region"],
                                               owner=row["owner"], sample=row["sample"], reasons=row["failures"]))
            pending = snapshot["pending"]
            failed += [dict(call=snapshot["call"], reason=reason) for reason in pending["failures"]]
            failed += [dict(call=snapshot["call"], kind="pending", layer=row["layer"], reasons=row["failures"])
                       for row in pending["banks"] if row["failures"]]
        output = self.output.report(state)
        enough = bool(self.evicted_owners) if require_kv_eviction else True
        return dict(passed=not failed and not self.ledger.failure_count and not self.output_time_failure_count and enough and output["finite_outputs"] and output["finite_private_state"],
                    structural_failures=failed, write_ledger_failures=self.ledger.failures,
                    write_ledger_failure_count=self.ledger.failure_count, output=output,
                    output_time_checks=self.calls, output_time_failure_count=self.output_time_failure_count,
                    output_time_failures=self.output_time_failures,
                    require_kv_eviction=require_kv_eviction, required_eviction_observed=enough,
                    kv_owner_samples_observed=len(self.owner_keys), kv_owner_samples_ever_full=len(self.ever_full),
                    kv_owner_samples_with_verified_eviction=len(self.evicted_owners),
                    eviction_witnesses=len(self.evictions), snapshots=self.snapshots,
                    positive_writes=sum(self.ledger.write_counts.values()),
                    maximum_retained_oracle_timestamps=self.ledger.maximum_retained_timestamps,
                    oracle="actual positive prepare receptions, original int64 times, per-owner window-bounded queues; no K/V payload copy or whole-stream trace",
                    eviction_definition="previously positive earliest retained time advances at two sampled boundaries; retained write set independently checked",
                    inactive_owner_policy="empty and inactive owners are valid; age is relative to their last positive write, not current global clock")
