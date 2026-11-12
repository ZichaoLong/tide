"""Numerical execution identity for typed KV columns, separate from mathematics."""

POSITION_POLICY = "bounded-common-origin-v1"


def validate_column_position_record(record):
    """Refuse an old numerical policy before constructing a model or device owner."""
    effective = record.get("effective", {})
    if record.get("schema") == "tide-weighted-execution-v5":
        raise ValueError("execution-v5 column positions require the original fixed source; "
                         "explicitly rebuild execution-v6 from requested configuration to adopt bounded-common-origin-v1")
    if (record.get("schema") == "tide-weighted-execution-v6"
            and effective.get("position_policy") != POSITION_POLICY):
        raise ValueError("execution-v6 position_policy identity mismatch")
