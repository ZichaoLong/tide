"""Keep planning, execution and recovery on the same observable boundary."""


def needs_trace(execution, task, *, reference=False):
    if (reference or "record_policy" not in execution
            or execution.get("return_mode", "complete") == "complete"):
        return True
    plugin = task.get("plugin")
    if not plugin:
        return False
    # Only a current explicit declaration can remove application-visible roots.
    # A frozen pre-declaration execution keeps its original complete records.
    return (plugin.get("observations", "complete") != "outputs"
            or not execution["record_policy"].startswith("selected-boundary-v2;"))
