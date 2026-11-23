"""Place unopened A trajectories on available CPU or NPU capacity."""
from .dispatch_placement3 import DEVICES, UNBOUND
from .prepare_resume3 import reference
from .round3 import TUNING_SEEDS
from .scheduler3 import freeze_json, read
from .source_migration3 import read_bound

POLICY = "stage-a-available-resources-v1"


def members(task):
    if task["phase"] == "a1":
        return [f'{task["model_id"]}-a1-{seed}-{task["recipe_id"]}' for seed in TUNING_SEEDS]
    return [task["id"]]


class ResourcePlacement:
    policy = POLICY

    def __init__(self, campaign):
        self.campaign = campaign
        self.reassigned = {}
        self.groups = {}

    def prepare_task(self, task):
        if task["phase"] not in ("init", "a1"):
            return
        batch = self.campaign.migration
        ids = members(task)
        fixed = set()
        for tid in ids:
            old = batch["completed"].get(tid) or batch["outstanding"].get(tid)
            if old and (tid in batch["completed"] or old["disposition"] != "not_started" or old.get("receipt")):
                fixed.add(read_bound(old["task"])["device"])
        if fixed and (len(fixed) != 1 or not fixed.issubset(DEVICES)):
            raise ValueError("incompatible previous devices for the same recipe: " + ids[0])
        device = next(iter(fixed), UNBOUND)
        if device == UNBOUND:
            self.groups[task["id"]] = ids
        task["device"] = device
        old = batch["outstanding"].get(task["id"])
        if old and old["disposition"] == "not_started" and not old.get("receipt"):
            self.reassigned[task["id"]] = old["task"], device

    def decision_path(self, ids):
        return self.campaign.output / "decisions" / (ids[0] + "-device.json")

    def register(self, task, row):
        tid = task["id"]
        directory = self.campaign.output / "tasks" / tid
        if task["device"] == UNBOUND:
            ids = self.groups[tid]
            intent = freeze_json(directory / "intent.json", task)
            path = self.decision_path(ids)
            if not path.exists():
                if row.get("attempts"):
                    raise ValueError("started direct-placement task lacks its decision: " + tid)
                row.update(task_path=str(intent), device=UNBOUND)
                return
            decision = read(path)
            if (decision["policy"] != POLICY or set(decision["intents"]) != set(ids)
                    or decision["intents"][tid] != reference(intent) or decision["device"] not in DEVICES):
                raise ValueError("direct placement differs from its frozen intent: " + tid)
            task["device"] = decision["device"]
        path = freeze_json(directory / "task.json", task)
        row.update(task_path=str(path), device=task["device"])

    def bind(self, task, device, pending):
        ids = self.groups[task["id"]]
        rows = self.campaign.supervisor.state["tasks"]
        intents = {}
        for tid in ids:
            member, row = pending[tid], rows[tid]
            if member["device"] != UNBOUND or row["status"] != "pending" or row.get("attempts"):
                raise ValueError("direct placement requires an unopened trajectory: " + tid)
            ref = reference(row["task_path"])
            if read_bound(ref) != member:
                raise ValueError("changed direct-placement intent: " + tid)
            intents[tid] = ref
        if device not in DEVICES:
            raise ValueError("unavailable execution backend")
        freeze_json(self.decision_path(ids), {"policy": POLICY, "device": device, "intents": intents})
        for tid in ids:
            self.register(pending[tid], rows[tid])


def split_calibrations(outstanding):
    """Retain the original requests and recovery evidence for retired probes."""
    retired = {}
    for tid, row in list(outstanding.items()):
        task = read_bound(row["task"])
        if task.get("phase") == "preflight" and task.get("purpose") == "capacity_calibration":
            if task.get("evaluation_budget_key") or task["id"] != tid:
                raise ValueError("invalid calibration retirement: " + tid)
            retired[tid] = outstanding.pop(tid)
    return retired
