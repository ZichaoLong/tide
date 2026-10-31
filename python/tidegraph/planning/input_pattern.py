"""Static physical input patterns, independent of a graph execution trace.

Global logical positions determine times; per-port occurrence counters remain
contiguous even when a port has no input at a particular global position.
"""

DENSE_POLICY = "unit-strength; shared tensor per sample/position; sin content; physical ports separate"
STAGGERED_POLICY = ("staggered-v1; unit-strength; sample lengths P-(sample%min(3,P)); "
                    "omit (sample+global_position+port)%4==0 when multiple ports; "
                    "contiguous per-port occurrences; original logical times; shared sin content")


def policy(pattern):
    if pattern is None:
        return DENSE_POLICY
    if type(pattern) is str and pattern == "staggered-v1":
        return STAGGERED_POLICY
    raise ValueError("unknown explicit input pattern")


def coordinates(plan):
    """Yield sample, global logical position, physical port, occurrence index."""
    pattern = plan.get("input_pattern")
    policy(pattern)
    ports, positions = len(plan["graph"]["inputs"]), plan["positions"]
    for sample in range(plan["samples"]):
        length = positions if pattern is None else positions-sample % min(3, positions)
        occurrences = [0]*ports
        for position in range(length):
            for port in range(ports):
                if pattern is not None and ports > 1 and (sample+position+port) % 4 == 0:
                    continue
                yield sample, position, port, occurrences[port]
                occurrences[port] += 1
