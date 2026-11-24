"""Fresh physical measurements and per-request sharing permission for runq."""
from __future__ import annotations

from pathlib import Path
import time

from .occupied_npu3 import boot_id, resident_identity
from .shared_npu3 import load_queue_helper

MIB = 2**20
FRESH_SECONDS = 15


def process_unit(pid):
    try:
        for line in Path(f"/proc/{pid}/cgroup").read_text().splitlines():
            _, hierarchy, path = line.split(":", 2)
            if hierarchy in ("name=systemd", ""):
                return Path(path).name
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
    return None


def observe(helper=None, mapping=None):
    helper = helper or load_queue_helper()
    mapping = mapping or helper.parse_mapping(helper.run_npu_smi("info", "-m"))
    raw = helper.run_npu_smi("info")
    if "Process id" not in raw:
        raise ValueError("NPU driver process table unavailable")
    devices = helper.parse_status(raw, mapping)
    memory = {}
    for line in raw.split("Process id", 1)[1].splitlines():
        columns = [part.strip() for part in line.split("|")]
        if len(columns) >= 6 and columns[2].isdigit() and columns[4].isdigit():
            card = mapping[tuple(map(int, columns[1].split()))]
            memory[card, int(columns[2])] = int(columns[4]) * MIB
    cards = {}
    for card, status in devices.items():
        processes = []
        for pid in status.process_ids:
            identity = resident_identity(pid)
            processes.append({"pid": pid, "identity": identity, "unit": process_unit(pid),
                              "bytes": memory.get((card, pid), 0)})
        cards[str(card)] = dict(card=card, health=status.health, core=status.aicore_percent,
                                used=status.hbm_used_mib*MIB, total=status.hbm_total_mib*MIB, processes=processes)
    return {"at": time.time(), "boot": boot_id(), "cards": cards}


def external(card, owned):
    return [item["identity"] for item in card["processes"] if item["unit"] not in owned or not item["unit"]]


def grant_for(sample, target, units, until):
    card = sample["cards"].get(target.split(":", 1)[1])
    if card is None:
        raise ValueError("physical device is absent from driver mapping")
    residents = external(card, units)
    if None in residents:
        raise ValueError("a device occupant changed during identity capture; submit again")
    return {"card": card["card"], "boot": sample["boot"], "residents": residents, "until": until, "at": sample["at"]}


def covers(grant, observed, card, boot, now):
    return (grant and grant["card"] == card and grant["boot"] == boot and now < grant["until"]
            and all(p is not None and p in grant["residents"] for p in observed))


def free_bytes(card, jobs):
    # Only credit allocations attributed by the driver to a live runq unit.
    # Subtract the unallocated remainder of each peak from physical free HBM.
    actual = {}
    for process in card["processes"]:
        name = process["unit"]
        actual[name] = actual.get(name, 0) + process["bytes"]
    return card["total"] - card["used"] - sum(max(0, j["data"]["hbm"] - actual.get(j["unit"], 0)) for j in jobs)


def admission(samples, card_id, jobs, size, grants, *, opening=False, manual=False, now=None):
    now = time.time() if now is None else now
    if not samples or now - samples[-1]["at"] > FRESH_SECONDS:
        return "waiting for fresh NPU measurements"
    names = {j["unit"] for j in jobs}
    key = str(card_id)
    previous = None
    for sample in samples[-2:]:
        card = sample["cards"].get(key)
        if not card or card["health"] != "OK":
            return "device absent or unhealthy"
        others = external(card, names)
        if (others and sample is samples[-1] and not opening and
                (previous != others or len(samples)<2 or now-samples[-2]["at"] > FRESH_SECONDS)):
            return "waiting for two stable shared-device samples"
        previous = others
        if others:
            if not any(covers(g, others, card_id, sample["boot"], now) for g in grants):
                return "external occupants need a matching manual grant"
        elif opening and not manual and (card["core"] > 5 or card["used"] > 4096*MIB):
            return "waiting for an idle cooperative device"
    if free_bytes(samples[-1]["cards"][key], jobs) < size:
        return "waiting for peak HBM"
    return None


def install_predicate(helper, card_id, grant, size, manual):
    # Each lease loads a private copy. Both samples and locks are still taken
    # by the published helper's unmodified queue algorithm.
    base = helper.DeviceStatus
    parse = helper.parse_status
    evidence = []

    def parse_status(raw, mapping):
        if "Process id" not in raw:
            raise ValueError("NPU driver process table unavailable")
        return parse(raw, mapping)

    class Card(base):
        def is_free(self, *, max_hbm_mib, max_aicore_percent):
            if self.physical_id != card_id or self.health != "OK" or (self.hbm_total_mib-self.hbm_used_mib)*MIB < size:
                return False
            residents = [resident_identity(pid) for pid in self.process_ids]
            evidence.append(dict(at=time.time(), residents=residents, used_mib=self.hbm_used_mib, core=self.aicore_percent))
            if residents != evidence[0]["residents"]:
                return False
            if manual:
                return not residents or covers(grant, residents, card_id, boot_id(), time.time())
            return super().is_free(max_hbm_mib=max_hbm_mib, max_aicore_percent=max_aicore_percent)

    helper.DeviceStatus, helper.parse_status = Card, parse_status
    return evidence
