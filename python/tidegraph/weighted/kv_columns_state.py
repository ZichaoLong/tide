"""KV §1.2: owned ragged raw columns; exact decoding never replays updates."""
from dataclasses import dataclass
import torch
from ..records import State
from ..history import increment, int64
from .column_layout import SequenceLayout
from .tensors import norm


@dataclass(frozen=True)
class KVStateColumns:
    layout: SequenceLayout
    keys: torch.Tensor
    values: torch.Tensor
    summaries: torch.Tensor
    weights: torch.Tensor
    inert: torch.Tensor
    last_times: torch.Tensor
    observations: torch.Tensor

    @classmethod
    def from_states(cls, states):
        if not states:
            raise ValueError("KV columns require a nonempty sample domain")
        offsets = [0]
        for state in states:
            if not int64(state.last_time) or not int64(state.observations) or state.last_time < -1 or state.observations < 0:
                raise ValueError("KV column state requires exact int64 clocks/counts")
            offsets.append(offsets[-1]+len(state.slots["times"]))
        layout = SequenceLayout(torch.tensor(offsets, dtype=torch.int64),
                                torch.cat([s.slots["times"].detach().cpu() for s in states]))
        fields = {name: torch.cat([s.slots[name] for s in states]) for name in ("keys", "values", "summaries", "weights")}
        return cls(layout, **fields, inert=torch.stack([s.value for s in states]),
                   last_times=torch.tensor([s.last_time for s in states], dtype=torch.int64),
                   observations=torch.tensor([s.observations for s in states], dtype=torch.int64))

    def states(self):
        """Boundary materialization of actual raw fields, one record per sample."""
        stamps = self.layout.times.to(self.keys.device)
        return [State(self.inert[b], int(self.last_times[b]), int(self.observations[b]),
                      dict(times=stamps[a:z], keys=self.keys[a:z], values=self.values[a:z],
                           summaries=self.summaries[a:z], weights=self.weights[a:z]))
                for b, (a, z) in enumerate(self.layout.spans)]

    def detach(self):
        from dataclasses import replace
        return replace(self, **{name: getattr(self, name).detach() for name in
                       ("keys", "values", "summaries", "weights", "inert")})

    def validate(self, config):
        n, b = len(self.layout.times), self.layout.samples
        shapes = dict(keys=(n, config.width), values=(n, config.width), summaries=(n, config.summary_width),
                      weights=(n,), inert=(b, config.width+1))
        for name, shape in shapes.items():
            x = getattr(self, name)
            if x.shape != shape or (x.dtype, x.device) != (self.keys.dtype, self.keys.device):
                raise ValueError("KV state column metadata mismatch: "+name)
            if not x.is_floating_point() or not torch.isfinite(x).all():
                raise ValueError("KV state columns require finite floating fields")
        for t in (self.last_times, self.observations):
            if t.shape != (b,) or t.dtype != torch.int64 or t.device.type != "cpu":
                raise ValueError("KV state clocks/counts must be host int64 B")
        if ((self.last_times < -1).any() or (self.observations < self.layout.offsets.diff()).any()
                or ((self.observations == 0) != (self.last_times == -1)).any()
                or c_window_exceeded(config.window, self.layout.offsets.diff())):
            raise ValueError("KV state clock/count outside domain")
        if ((self.weights < 0) | (self.weights > 1)).any() or self.inert.count_nonzero():
            raise ValueError("invalid raw KV state values")
        for owner, (a, z) in enumerate(self.layout.spans):
            if z > a and self.layout.times[z-1] > self.last_times[owner]:
                raise ValueError("KV entry time exceeds last write time")


@dataclass(frozen=True)
class KVWriteBank:
    """Query storage for a window, including entries expired by its terminal cut."""
    layout: SequenceLayout
    keys: torch.Tensor
    values: torch.Tensor
    summaries: torch.Tensor
    weights: torch.Tensor

    @classmethod
    def from_state(cls, state):
        return cls(state.layout, state.keys, state.values, state.summaries, state.weights)


@dataclass(frozen=True)
class KVWindowColumns:
    layout: SequenceLayout
    mass: torch.Tensor
    content: torch.Tensor
    bank: KVWriteBank
    terminal: KVStateColumns


def _prepare_inputs(memory, layout, mass, content, left):
    """Validate raw state/clocks before any grouped floating preparation."""
    c = memory.config
    left.validate(c)
    if layout.samples != left.layout.samples or content.shape != (len(layout.times), c.width) or mass.shape != content.shape[:1]:
        raise ValueError("KV window row/sample domain mismatch")
    if (mass.dtype, mass.device) != (content.dtype, content.device) or (content.dtype, content.device) != (left.keys.dtype, left.keys.device):
        raise ValueError("KV window tensor metadata mismatch")
    if (mass < 0).any() or not torch.isfinite(mass).all() or not torch.isfinite(content).all():
        raise ValueError("KV window requires finite nonnegative received mass")
    positive = (mass.detach() > 0).cpu()
    for b, (a, z) in enumerate(layout.spans):
        if z > a and layout.times[a] <= left.last_times[b]:
            raise ValueError("KV column windows must advance the raw logical clock")
    write_rows = positive.nonzero().flatten()
    return positive, write_rows


def _assemble(memory, layout, mass, content, left, positive, write_rows, fresh):
    """Arrange actual writes and retain raw state; no projection or replay."""
    c = memory.config
    if not len(write_rows):
        return KVWindowColumns(layout, mass, content, KVWriteBank.from_state(left), left)
    permutation, offsets, times = [], [0], []
    lasts, counts = left.last_times.tolist(), left.observations.tolist()
    fresh_offset, old_count = 0, len(left.layout.times)
    for b, ((a, z), (old_a, old_z)) in enumerate(zip(layout.spans, left.layout.spans)):
        actual = layout.times[a:z][positive[a:z]]
        if len(actual) and actual[0] <= lasts[b]:
            raise ValueError("KV column writes must advance the raw logical clock")
        permutation.extend(range(old_a, old_z))
        permutation.extend(range(old_count+fresh_offset, old_count+fresh_offset+len(actual)))
        offsets.append(offsets[-1]+old_z-old_a+len(actual))
        times.extend((left.layout.times[old_a:old_z], actual))
        fresh_offset += len(actual)
        if len(actual):
            lasts[b] = int(actual[-1]); counts[b] = increment(counts[b], len(actual))
    index = torch.tensor(permutation, dtype=torch.int64, device=content.device)
    bank_layout = SequenceLayout(torch.tensor(offsets, dtype=torch.int64), torch.cat(times))
    fields = {name: torch.cat((getattr(left, name), x)).index_select(0, index) for name, x in fresh.items()}
    bank = KVWriteBank(bank_layout, **fields)
    terminal = KVStateColumns(bank_layout, **fields, inert=left.inert,
        last_times=torch.tensor(lasts, dtype=torch.int64), observations=torch.tensor(counts, dtype=torch.int64))
    if c.window:
        # Expiry occurs only on a positive write. All-zero receives retain old
        # expired entries exactly; a visible-state projection is not a codec.
        retained, offsets = [], [0]
        for b, (a, z) in enumerate(bank.layout.spans):
            ids = torch.arange(a, z)
            if counts[b] != int(left.observations[b]):
                ids = ids[(lasts[b]-bank.layout.times[a:z]) < c.window]
            retained.append(ids); offsets.append(offsets[-1]+len(ids))
        ids = torch.cat(retained)
        trimmed = {name: x.index_select(0, ids.to(x.device)) for name, x in fields.items()}
        terminal = KVStateColumns(SequenceLayout(torch.tensor(offsets, dtype=torch.int64), bank.layout.times[ids]),
            **trimmed, inert=left.inert, last_times=terminal.last_times, observations=terminal.observations)
    return KVWindowColumns(layout, mass, content, bank, terminal)


def prepare_kv_columns(memory, layout, mass, content, left):
    """Append actual positive writes across B/L, retaining exact raw metadata."""
    positive, write_rows = _prepare_inputs(memory, layout, mass, content, left)
    fresh = _project_writes(memory, mass, content, write_rows)
    return _assemble(memory, layout, mass, content, left, positive, write_rows, fresh)


def _project_writes(memory, mass, content, write_rows):
    if not len(write_rows):
        return None
    writes = write_rows.to(content.device)
    normalized = norm(content.index_select(0, writes), memory.config.epsilon)
    kv = normalized@memory.base.t()+(normalized@memory.adapter_v.t())@memory.adapter_u.t()
    key, value = kv.chunk(2, -1)
    return dict(keys=key, values=value, summaries=normalized@memory.summary.t(), weights=mass[writes].clamp_max(1))


def preparation_group_bytes(memory, rows, element_size):
    """Conservative packing/adjoint storage for four simultaneous block maps.

    Each linear scope reserves two matrix copies, three input copies, two
    outputs; normalized/gathered content and the KV sum are counted separately.
    Public per-node state/query budgets continue to apply outside this scope.
    """
    c = memory.config
    weights = 2*c.width*c.width+3*c.width*c.adapter_rank+c.width*c.summary_width
    values = rows*(25*c.width+5*c.adapter_rank+2*c.summary_width)
    return (2*weights+values)*max(element_size, 4)+16*rows


def prepare_kv_columns_many(requests, *, max_workspace_bytes):
    """Node groups of actual equal-sized write columns, independent raw states.

    Requests are (memory, layout, mass, content, left). Grouping follows shape,
    dtype and device, never node name or a predicted activity trace. The byte
    ceiling limits block packing; a singleton uses the original unstacked path.
    """
    from .row_linear import linear_blocks
    if type(max_workspace_bytes) is not int or not 0 < max_workspace_bytes < 2**63:
        raise ValueError("KV preparation group budget must be a positive int64 byte count")
    prepared, groups, result = [], {}, [None]*len(requests)
    for i, request in enumerate(requests):
        memory, layout, mass, content, left = request
        positive, writes = _prepare_inputs(*request)
        prepared.append((positive, writes))
        if not len(writes):
            result[i] = _assemble(*request, positive, writes, None)
            continue
        c = memory.config
        compact = content.dtype in {torch.float32, torch.float64}
        key = (content.device, content.dtype, None if compact else len(writes),
               c.width, c.adapter_rank, c.summary_width, c.epsilon)
        groups.setdefault(key, []).append(i)

    def execute(ids):
        if len(ids) == 1:
            i = ids[0]
            memory, layout, mass, content, left = requests[i]
            positive, rows = prepared[i]
            fresh = _project_writes(memory, mass, content, rows)
            result[i] = _assemble(*requests[i], positive, rows, fresh)
            return
        if requests[ids[0]][3].dtype in {torch.float32, torch.float64}:
            from .kv_tensor_prepare import prepare
            for i, fresh in zip(ids, prepare(requests, prepared, ids)):
                result[i] = _assemble(*requests[i], *prepared[i], fresh)
            return
        memories = [requests[i][0] for i in ids]
        writes = [prepared[i][1].to(requests[i][3].device) for i in ids]
        normalized = [norm(requests[i][3].index_select(0, index), memory.config.epsilon)
                      for i, index, memory in zip(ids, writes, memories)]
        bases = linear_blocks(normalized, [m.base for m in memories])
        low_rank = linear_blocks(normalized, [m.adapter_v for m in memories])
        updates = linear_blocks(low_rank, [m.adapter_u for m in memories])
        summaries = linear_blocks(normalized, [m.summary for m in memories])
        for i, index, base, update, summary in zip(ids, writes, bases, updates, summaries):
            key, value = (base+update).chunk(2, -1)
            fresh = dict(keys=key, values=value, summaries=summary, weights=requests[i][2][index].clamp_max(1))
            result[i] = _assemble(*requests[i], *prepared[i], fresh)

    for ids in groups.values():
        batch = []
        for i in ids:
            candidate = [*batch, i]
            width = max(len(prepared[j][1]) for j in candidate)
            cost = sum(preparation_group_bytes(requests[j][0], width, requests[j][3].element_size()) for j in candidate)
            if batch and cost > max_workspace_bytes:
                execute(batch); batch = []
            batch.append(i)
        if batch:
            execute(batch)
    return result


def c_window_exceeded(window, counts):
    return bool(window and (counts > window).any())
