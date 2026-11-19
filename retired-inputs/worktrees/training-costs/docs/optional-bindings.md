# Optional current HB bindings and support coverage

The default [main menu](../math/instances/backbone-comparison.md) is unchanged.
The options below use the same fixed mathematics, parameter ownership and state
semantics; they do not enable old teaching instances or an arbitrary graph API.

## Fixed-pass Hub groups

`ModelConfig(hub_allocation="fixed_pass", ...)` selects the additional group
binding in [HB §3](../math/instances/hb-multiscale-weighted-message.md#3-三种分区及角色预算):
`k_H=h`, `c_H=mu_H=min(1,2*received_H)`. It requires Exchange/Multiscale with
Hubs and `partition="group"`. Ordinary regions keep their selected allocation.
The H local description/score and regional scoring parameters are omitted as
required by [parameter profiles](../math/instances/parameter-profiles.md).
Reception, all positive writes, private KV/SSD state and Full remain enabled.
`bounded` retains the existing `k_H=ceil(h/2)` role-quad binding.
This is different from changing to singleton regions or raising the D budget.

## Explicit local clocks and edge-delay comparisons

The input schedule remains `TopologyConfig.period`. `ClockConfig` independently
declares the KV position divisor, logical KV window, short-summary decay rate
(shared by KV and SSD), and SSD logical convolution stride. With `None`, defaults
remain position divisor `period`, window `128*period`, short decay
`log(2)/(8*period)`, and stride `period` for Settle or 1 for Timed. Long-summary
decay and SSD passive matrix decay remain zero. All time metadata remains int64.
Settle stores at most `ceil(kv_window/period)` window slots; Timed can write at
every tick and stores up to `kv_window` slots. These are logical-time bounds,
not a count of selected Full actions.

```python
import math
from tide import ClockConfig, ModelConfig, TopologyConfig, edge_delay_pair

clock = ClockConfig(position_period=10, position_origin="absolute",
                    kv_window=1280, short_decay=math.log(2)/80, ssd_stride=1)
base = ModelConfig(topology=TopologyConfig(kind="multiscale", profile="compact-9",
                                         n=8, hubs=4), memory="X", clock=clock)
settle_config, timed_config = edge_delay_pair(base, period=10)
```

`edge_delay_pair` requires Multiscale, a common input period greater than the
number of layers, and every local clock explicitly set. `absolute` declares the
same `Pos(theta)=theta/position_period` on both sides. The default `binding`
declares the original rank/input-origin offsets; each node-constant offset
cancels in the relative KV rotations, which subtract int64 times before conversion.
The strict pair changes edge delays only. It does **not** promise identical
arrivals, support, writes, state, outputs or output-slot layouts: those are the
mathematical effects under comparison. The selected clock remains fixed through
all continued windows and is serialized with checkpoints.

## Optional terminal Hubs

`TopologyConfig(terminal_hubs=True, ...)` explicitly selects the extra graph
defined by the current [Multiscale](../math/topologies/hb-multiscale.md) and
[Exchange](../math/topologies/hb-exchange.md) definitions. It requires configured
Hubs; the existing mask also applies on the final layer. Outputs remain D ports.
Group/singleton terminal H has no path to D output, but adds parameters/state;
mixed terminal H participates in the final joint allocation and can change D
output. The default graph still deletes terminal Hubs. This option does not
change the main-menu catalog counts.

## Machine-verifiable coverage

Run with the private Python environment, without allocating any model:

```sh
python -m scripts.support_catalog --output NEW-support.json
```

The generator checks all 41 spatial/55 temporal profiles, 475 geometry settings,
225,400 parameter configurations and 676,200 allocation rows against the fixed
menu, including widths, native periods, probes and legal D budgets. It records
source-file hashes. The independent catalog's rank 0 marks unused adapters;
the model API uses an ignored valid rank for independent parameters.

The JSON keeps four qualification categories separate. Metadata acceptance is
reported by the generator. It does not run neural execution, numerical checks
or performance measurements, and it does not elevate older records to a new
source. `tests/test_optional_bindings.py` supplies finite independent topology,
parameter-count, value/VJP, clock, multi-update and actual restore-update
witnesses. Other module tests retain their original thresholds. Device/precision
qualification and performance remain scoped to their actual recorded inputs;
the optional choices are not part of the old CPU/NPU performance matrices.
