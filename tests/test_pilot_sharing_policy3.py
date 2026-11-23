"""Measured concurrency decisions without devices or wall-clock sleeps."""
from pilotlab.sharing_policy3 import SharingPolicy


def observation(tokens, status="training", age=0):
    return {"tokens": tokens, "status": status, "age_seconds": age}


def test_growth_keeps_original_cohort_and_reduces_only_future_admission():
    policy = SharingPolicy({"stable_window_seconds": 10, "min_gain_ratio": .1}, 16)
    policy.observe({"a": observation(0)}, [], now=0)
    policy.observe({"a": observation(100)}, [], now=10)
    assert policy.limit == 2 and policy.baseline["worker_ids"] == ["a"]
    policy.observe({"a": observation(110), "b": observation(0, "initial_checks")}, [], now=11)
    assert policy.baseline["worker_ids"] == ["a"]
    policy.observe({"a": observation(120), "b": observation(10)}, [], now=12)
    policy.observe({"a": observation(170), "b": observation(60)}, [], now=22)
    assert policy.limit == 1
    event = policy.snapshot()["recent_events"][-1]
    assert event["gain_ratio"] == 0 and event["total_tokens_per_second"] == 10
    assert event["reason"] == "aggregate_throughput_gain_below_threshold"
    policy.observe({"a": observation(175), "b": observation(65)}, [], now=23)
    assert policy.snapshot()["reason"] == "healthy_workers_draining_above_new_limit"


def test_good_aggregate_growth_advances_only_one_level_per_stable_window():
    policy = SharingPolicy({"stable_window_seconds": 1}, 16)
    rows = {"0": observation(0)}
    policy.observe(rows, [], now=0)
    for count in (1, 2, 4, 8):
        rows = {str(index): observation(100) for index in range(count)}
        policy.observe(rows, [], now=count*2)
        rows = {str(index): observation(200) for index in range(count)}
        policy.observe(rows, [], now=count*2+1)
        assert policy.limit == count*2
    assert policy.snapshot()["levels"] == [1, 2, 4, 8, 16]


def test_reference_exit_evaluation_or_external_identity_change_invalidates_comparison():
    for change in ("exit", "evaluation", "external"):
        policy = SharingPolicy({"stable_window_seconds": 10}, 8)
        policy.observe({"a": observation(0)}, [], now=0)
        policy.observe({"a": observation(100)}, [], now=10)
        rows = {"a": observation(110), "b": observation(10)}
        external = []
        if change == "exit":
            rows.pop("a")
        elif change == "evaluation":
            rows["a"]["status"] = "saving_checkpoint"
        else:
            external = [(9999, 33, 1000)]
        policy.observe(rows, external, now=11)
        assert policy.baseline is None
        assert not any(event["kind"] == "window" and "gain_ratio" in event for event in policy.history)


def test_stale_regressing_and_nonprogressing_samples_never_raise_limit():
    policy = SharingPolicy({"stable_window_seconds": 1, "heartbeat_max_age_seconds": 2}, 16)
    policy.observe({"a": observation(100)}, [], now=0)
    policy.observe({"a": observation(200, age=3)}, [], now=1)
    policy.observe({"a": observation(200)}, [], now=2)
    policy.observe({"a": observation(200)}, [], now=3)
    policy.observe({"a": observation(200)}, [], now=4)
    policy.observe({"a": observation(100)}, [], now=5)
    assert policy.limit == 1


def test_resource_pressure_lowers_future_limit_without_rewriting_running_workers():
    policy = SharingPolicy({"stable_window_seconds": 1}, 16)
    policy.observe({"a": observation(0)}, [], now=0)
    policy.observe({"a": observation(100)}, [], now=1)
    active = {"a": observation(110)}
    policy.resource_wait("hbm_reservations_full", active, now=2)
    assert policy.limit == 1 and active == {"a": observation(110)}
    assert policy.snapshot()["reason"] == "hbm_reservations_full"
    policy.backlog(True, True, now=2.5)
    assert policy.ceiling == 1
    policy.backlog(True, True, now=3)
    assert policy.ceiling == 16
    assert policy.limit == 1  # Recovery permits a measured retrial, not a blind jump.


def test_short_workloads_get_only_one_bounded_unmeasured_trial_with_progress():
    policy = SharingPolicy({"stable_window_seconds": 10}, 16)
    policy.backlog(True, True, now=0)
    for now in range(1, 21):
        policy.backlog(True, True, now=now)
    assert policy.limit == 1  # No observed token progress.
    policy.backlog(True, False, completed_tokens=32, now=21)
    assert policy.limit == 1  # No resource headroom.
    policy.backlog(True, True, completed_tokens=32, now=22)
    assert policy.limit == 2 and policy.unmeasured_exploration_used
    assert policy.history[-1]["kind"] == "unmeasured_exploration"
    for now in range(30, 100):
        policy.backlog(True, True, completed_tokens=32, now=now)
    assert policy.limit == 2
