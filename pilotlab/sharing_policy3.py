"""Measured admission limits for a held card; never terminate healthy workers."""
from __future__ import annotations

import copy
import math
import time


class SharingPolicy:
    """Compare stable training windows while retaining the pre-growth cohort.

    Adding workers is the experiment, so it does not itself erase its baseline.
    Losing an original worker, its training state, or the external-occupant
    identity does erase it.  Lowering a limit only affects later admissions.
    """
    def __init__(self, config, max_parallel):
        self.levels = [value for value in (1, 2, 4, 8, 16) if value <= max_parallel]
        self.window_seconds = float(config.get("stable_window_seconds", 30.))
        self.max_age = float(config.get("heartbeat_max_age_seconds", 15.))
        self.min_gain = float(config.get("min_gain_ratio", .05))
        if not self.levels or not 0 < self.window_seconds <= 3600 or not 0 < self.max_age <= 3600:
            raise ValueError("invalid sharing measurement window/heartbeat age")
        if not math.isfinite(self.min_gain) or not 0 <= self.min_gain < 1:
            raise ValueError("min_gain_ratio must be finite in [0,1)")
        self.limit = self.levels[0]
        self.ceiling = self.levels[-1]
        self.baseline = None
        self.window = None
        self.external = None
        self.held_cohort = set()
        self.reason = "awaiting_first_stable_training_window"
        self.history = []
        self.events = []
        self.sequence = 0
        self.backlog_since = None
        self.last_backlog_seen = None
        self.progress_seen = False
        self.unmeasured_exploration_used = False
        self.resource_limited = False
        self.last_resource_wait = None
        self.resource_dispatch = config.get("resource_dispatch", False)
        if self.resource_dispatch:
            self.limit = self.ceiling = max_parallel
            self.reason = "admission_by_available_resources"

    def event(self, kind, reason, now, **details):
        self.reason = reason
        row = {"sequence": self.sequence, "observed_monotonic": now, "kind": kind,
               "reason": reason, "admission_limit": self.limit, **details}
        self.sequence += 1
        self.events.append(row)
        self.history = (self.history + [row])[-32:]

    def invalidate(self, reason, now, *, baseline=False):
        had_window = self.window is not None
        self.window = None
        if baseline:
            self.baseline = None
        if had_window or self.reason != reason:
            self.event("invalidated", reason, now)

    def observe(self, workers, external_identity, *, now=None):
        if self.resource_dispatch:
            return
        now = time.monotonic() if now is None else now
        external = tuple(tuple(row) for row in external_identity)
        current = set(workers)
        valid = {key for key, value in workers.items()
                 if value.get("status") == "training" and type(value.get("tokens")) is int
                 and value["tokens"] >= 0 and isinstance(value.get("age_seconds"), (int, float))
                 and 0 <= value["age_seconds"] <= self.max_age}
        self.progress_seen |= any(workers[key]["tokens"] > 0 for key in valid)
        if self.external is not None and external != self.external:
            self.invalidate("external_occupant_identity_changed", now, baseline=True)
            self.ceiling = self.levels[-1]
        self.external = external
        if self.held_cohort and not self.held_cohort <= current:
            self.ceiling, self.held_cohort = self.levels[-1], set()
        if self.baseline and not set(self.baseline["worker_ids"]) <= valid:
            self.invalidate("reference_cohort_left_training", now, baseline=True)
        if not current or valid != current:
            self.invalidate("workers_not_all_fresh_training", now)
            return
        if len(current) != self.limit:
            self.invalidate("waiting_for_admission_target" if len(current) < self.limit
                            else "healthy_workers_draining_above_new_limit", now)
            return
        tokens = {key: workers[key]["tokens"] for key in sorted(current)}
        if self.window is None or set(self.window["tokens"]) != current:
            if self.window is not None:
                self.invalidate("window_worker_composition_changed", now)
            self.window = {"started": now, "tokens": tokens}
            self.reason = "collecting_stable_training_window"
            return
        elapsed = now - self.window["started"]
        if elapsed < 0 or any(tokens[key] < old for key, old in self.window["tokens"].items()):
            self.invalidate("clock_or_token_counter_regressed", now, baseline=True)
            return
        if elapsed < self.window_seconds:
            return
        deltas = {key: tokens[key] - old for key, old in self.window["tokens"].items()}
        if any(delta <= 0 for delta in deltas.values()):
            self.invalidate("not_every_worker_made_training_progress", now, baseline=True)
            return
        rate = sum(deltas.values()) / elapsed
        measured = {"parallelism": len(current), "worker_ids": sorted(current), "tokens": sum(deltas.values()),
                    "elapsed_seconds": elapsed, "total_tokens_per_second": rate,
                    "external_conditions": external,
                    "worker_tokens_per_second": {key: delta / elapsed for key, delta in deltas.items()}}
        self.window = None
        baseline = self.baseline
        if baseline is not None:
            gain = rate / baseline["total_tokens_per_second"] - 1
            measured.update(reference_parallelism=baseline["parallelism"],
                            reference_worker_ids=baseline["worker_ids"], gain_ratio=gain)
            if gain < self.min_gain:
                self.limit = baseline["parallelism"]
                self.ceiling = self.limit
                self.held_cohort = current
                self.baseline = None
                self.event("window", "aggregate_throughput_gain_below_threshold", now, **measured)
                return
        next_level = next((value for value in self.levels if self.limit < value <= self.ceiling), None)
        if next_level is None:
            self.baseline = None
            self.event("window", "measured_limit_retained", now, **measured)
            return
        self.baseline = measured
        self.limit = next_level
        self.event("window", "stable_window_allows_next_concurrency_trial", now, **measured)

    def resource_wait(self, reason, active_ids, *, now=None):
        """A resource ceiling can lower future admissions, never running work."""
        if self.resource_dispatch:
            self.reason = reason
            return
        now = time.monotonic() if now is None else now
        self.last_resource_wait = now
        if self.limit <= 1:
            if self.reason != reason:
                self.event("resource_wait", reason, now)
            return
        lower = max(value for value in self.levels if value < self.limit)
        self.limit = self.ceiling = lower
        self.resource_limited = True
        self.held_cohort = set(active_ids)
        self.window = self.baseline = None
        self.event("resource_wait", reason, now)

    def backlog(self, present, resource_available, *, completed_tokens=0, now=None):
        """One explicitly unmeasured two-worker trial for short progressing jobs."""
        if self.resource_dispatch:
            return
        now = time.monotonic() if now is None else now
        self.progress_seen |= type(completed_tokens) is int and completed_tokens > 0
        if (resource_available and self.resource_limited and self.last_resource_wait is not None
                and now - self.last_resource_wait >= self.window_seconds):
            self.ceiling = self.levels[-1]
            self.resource_limited = False
            self.event("resource_recovered", "resource_headroom_recovered_allow_measured_retrial", now)
        if self.last_backlog_seen is not None and now - self.last_backlog_seen > self.window_seconds:
            self.backlog_since = None
        if not present:
            return
        self.last_backlog_seen = now
        self.backlog_since = now if self.backlog_since is None else self.backlog_since
        if (self.limit == 1 and self.ceiling >= 2 and not self.unmeasured_exploration_used
                and self.progress_seen and resource_available and now - self.backlog_since >= self.window_seconds):
            self.unmeasured_exploration_used = True
            self.limit = 2
            self.window = self.baseline = None
            self.event("unmeasured_exploration", "short_workload_backlog_with_progress_and_resource_headroom", now,
                       measured_gain=None, maximum_unmeasured_parallelism=2)

    def take_events(self):
        value, self.events = self.events, []
        return value

    def snapshot(self):
        if self.resource_dispatch:
            return {"schema": "pilot-r3-resource-sharing-v1", "admission_limit": self.limit,
                    "reason": self.reason, "max_parallel": self.ceiling}
        return {"schema": "pilot-r3-sharing-policy-v1", "admission_limit": self.limit,
                "growth_ceiling": self.ceiling, "levels": self.levels,
                "stable_window_seconds": self.window_seconds, "min_gain_ratio": self.min_gain,
                "heartbeat_max_age_seconds": self.max_age, "reason": self.reason,
                "unmeasured_exploration_used": self.unmeasured_exploration_used,
                "reference_window": copy.deepcopy(self.baseline), "events_observed": self.sequence,
                "recent_events": copy.deepcopy(self.history),
                "scope": "aggregate card training-token throughput from worker heartbeats; mixed workloads and shared contention are recorded",
                "reduction_policy": "limit subsequent admissions; never terminate healthy workers"}
