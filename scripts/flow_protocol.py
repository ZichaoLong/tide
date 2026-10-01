"""Continuous workload v2; v1 reset-window packets keep their original meaning."""
import hashlib
from flow_topology import canonical_bytes, make_packet, validate_packet as validate_v1


def make_continuous_packet(**kwargs):
    old = make_packet(**kwargs)
    packet = {k: v for k, v in old.items() if k != "sha256"}
    packet.update(schema="tide-complete-flow-workload-v2",
                  state_boundary="continuous state/history/pending/KV; explicit detach after each optimizer update",
                  parameter_policy="named-lcg31-v1; FP32 source values cast to payload dtype",
                  model_profile="edge-affine-lh-silu-rms-norm-fp32-v1",
                  loss_policy="next-token cross entropy; sum over present outputs / requested tokens",
                  input_policy="(absolute_position*7+sample*3)%vocab; embedding/head included")
    return dict(packet, sha256=hashlib.sha256(canonical_bytes(packet)).hexdigest())


def validate_packet(packet):
    if packet.get("schema") == "tide-complete-flow-workload-v1":
        return validate_v1(packet)
    if packet.get("schema") != "tide-complete-flow-workload-v2":
        raise ValueError("unknown continuous packet schema")
    payload = {k: v for k, v in packet.items() if k != "sha256"}
    if packet.get("sha256") != hashlib.sha256(canonical_bytes(payload)).hexdigest():
        raise ValueError("packet content hash mismatch")
    config = {k: v for k, v in packet["workload"].items() if k != "stride"}
    if make_continuous_packet(graph=packet["graph"], **config) != packet:
        raise ValueError("inconsistent continuous packet")
    return packet


def native_text(packet):
    validate_packet(packet)
    if packet["schema"] != "tide-complete-flow-workload-v2":
        from flow_topology import native_text as v1
        return v1(packet)
    g, c = packet["graph"], packet["workload"]
    rows = ["TIDE_COMPLETE_FLOW_2", packet["sha256"],
            " ".join(str(c[k]) for k in ("width", "batch", "tokens", "vocab", "seed")),
            c["memory"] + " " + g["kind"],
            " ".join(map(str, (g["nodes"], g["regions"], len(g["edges"]), len(g["inputs"]),
                                len(g["outputs"]), c["stride"], c["budget"], int(c["clear"])))),
            " ".join(map(str, g["node_regions"])), " ".join(map(str, g["ranks"])),
            " ".join(map(str, g["inputs"])), " ".join(map(str, g["outputs"]))]
    rows.extend(" ".join(map(str, edge)) for edge in g["edges"])
    return "\n".join(rows) + "\n"
