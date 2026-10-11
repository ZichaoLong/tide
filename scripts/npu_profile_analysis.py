"""Explicit TorchNPU 2.10 offline-analysis adapter, isolated from model execution.

One session and one parser task run at a time. The vendor NON_BLOCKING timeline
helper needs an explicit export dependency when concurrency is one; otherwise
the manager exits while dependent tasks are still queued. No installed files
or parser algorithms are changed. The caller must enforce process bounds.
"""
from contextlib import contextmanager
from copy import copy
import inspect
import json
from pathlib import Path

from durable_records import write_json
from source_identity import digest


@contextmanager
def serial_parser(parser, receipt):
    original = parser.ConcurrentTasksManager
    if 'max_concurrent_num' not in inspect.signature(original).parameters:
        raise RuntimeError('unsupported TorchNPU analysis manager interface')

    class SerialManager(original):
        def __init__(self, **kwargs):
            kwargs['max_concurrent_num'] = 1
            super().__init__(**kwargs)

        def add_task(self, task):
            task = copy(task)
            if task.mode & 4:  # TorchNPU 2.10 ConcurrentMode.NON_BLOCKING
                if type(task).__name__ != 'CANNTimelineParser' or 'export' not in self.task_infos:
                    raise RuntimeError('unsupported nonblocking analysis task')
                task.mode &= ~4
                task.deps = list(dict.fromkeys([*task.deps, 'export']))
            if task.mode not in (1, 2):  # SUB_PROCESS, PTHREAD
                raise RuntimeError('unsupported analysis task mode')
            super().add_task(task)

        def run(self):
            try:
                super().run()
            finally:
                tasks = {name: info.status.name for name, info in self.task_infos.items()}
                passed = bool(tasks) and all(s == 'Succeed' for s in tasks.values())
                write_json(receipt, dict(state='passed' if passed else 'failed', tasks=tasks))
            if not passed:
                raise RuntimeError('offline analysis left failed or unexecuted tasks')

    parser.ConcurrentTasksManager = SerialManager
    try:
        yield
    finally:
        parser.ConcurrentTasksManager = original


def analyse_raw(raw, output):
    # Lazy import keeps help and CPU contract tests vendor independent.
    import torch_npu
    from torch_npu.profiler.profiler import analyse
    from torch_npu.profiler.analysis import _profiling_parser as parser
    from torch_npu.profiler.analysis.prof_common_func import _task_manager

    if torch_npu.__version__.split('+')[0] != '2.10.0':
        raise RuntimeError('serial offline adapter requires qualified TorchNPU 2.10.0')
    receipt = output/'analysis-tasks.json'
    if receipt.exists():
        raise FileExistsError(receipt)
    identity = dict(torch_npu=torch_npu.__version__, session_workers=1, parser_workers=1,
        parser_sha256=digest(Path(parser.__file__)),
        task_manager_sha256=digest(Path(_task_manager.__file__)),
        adapter='serial tasks; blocking timeline after export; explicit text output')
    write_json(output/'analysis-adapter.json', identity)
    with serial_parser(parser, receipt):
        analyse(str(raw), max_process_number=1, export_type='text')
    # TorchNPU may swallow worker exceptions. A trace alone cannot certify that
    # the complete requested parser inventory ran successfully.
    if not receipt.is_file() or json.loads(receipt.read_text()).get('state') != 'passed':
        raise RuntimeError('offline analysis did not finish every parser task')
    return identity
