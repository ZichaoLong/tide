"""Explicit publication cadence; execution and optimizer windows are unchanged."""


def publication_plan(measurement, start_window=0):
    interval = measurement.get("checkpoint_interval", 1)
    total = measurement["warmup"]+measurement["steps"]
    end = start_window+total
    count = end//interval-start_window//interval+int(end % interval != 0)
    # Preserve the existing default reservation, even for a one-window run.
    generations = 2 if interval == 1 else min(2, count)
    return dict(interval=interval, publications=count, simultaneous_generations=generations,
                final_next_window=end, final_always_saved=True,
                recovery_scope="only published complete boundaries; unsaved windows are lost on interruption")


def due(plan, next_window):
    return next_window == plan["final_next_window"] or next_window % plan["interval"] == 0
