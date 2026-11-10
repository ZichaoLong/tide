"""Keep planning, execution and recovery on the same observable boundary."""


def needs_trace(execution, task, *, reference=False):
    return (reference or "record_policy" not in execution
            or execution.get("return_mode", "complete") == "complete"
            or bool(task.get("plugin")))
