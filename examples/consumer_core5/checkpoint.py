"""Whole-application boundary, additional to Tide's graph-only checkpoint."""
from pathlib import Path
import tempfile
import torch
from tidegraph.checkpoint import save, load
from recording import application_identity, fingerprint, snapshot, tensor_record

SCHEMA = 'example-core5-application-v1'


def identity(task, dependency):
    return dict(application=application_identity(), workload_id=task.workload_id,
                execution_id=fingerprint(task.application_execution), dependency=dependency['lock_sha256'])


def save_application(path, task, dependency):
    # The graph codec owns graph weights, aliases, raw state and pending.
    # The application owns a combined graph/head optimizer and all RNG/progress.
    with tempfile.TemporaryDirectory() as temporary:
        graph = Path(temporary)/'graph.pt'
        save(graph, task.binding.graph, task.binding.model, task.continuation)
        record = dict(schema=SCHEMA, identity=identity(task, dependency), graph=graph.read_bytes(),
            head=snapshot(task.head.state_dict()), optimizer=snapshot(task.optimizer.state_dict()),
            scheduler=snapshot(task.scheduler.state_dict()), parameter_names=list(task.named_parameters()),
            step=task.step_index, position=task.step_index*task.positions, data_rng=task.data_rng.get_state(),
            target_rng=task.target_rng.getstate(), torch_rng=torch.get_rng_state())
    # One atomic no-overwrite file, including graph bytes. For this small starter
    # only: large experiments should use a separately designed sharded bundle.
    tensor_record(path, record)


def restore_application(path, task, dependency):
    """Load into a newly constructed owner, before publishing any run directory."""
    r = torch.load(path, map_location='cpu', weights_only=True)
    if r.get('schema') != SCHEMA or r.get('identity') != identity(task, dependency):
        raise ValueError('application checkpoint identity mismatch')
    if r.get('parameter_names') != list(task.named_parameters()):
        raise ValueError('application parameter ownership mismatch')
    step = r.get('step')
    if type(step) is not int or step < 0 or type(r.get('position')) is not int or r['position'] != step*task.positions:
        raise ValueError('application progress mismatch')
    if not isinstance(r.get('graph'), bytes):
        raise ValueError('application graph checkpoint missing')
    with tempfile.TemporaryDirectory() as temporary:
        graph = Path(temporary)/'graph.pt'
        graph.write_bytes(r['graph'])
        q = load(graph, task.binding.graph, task.binding.model)
    if q.cut != step*task.span or q.batch_size != task.batch:
        raise ValueError('graph/application clock mismatch')
    task.head.load_state_dict(r['head'], strict=True)
    task.optimizer.load_state_dict(r['optimizer'])
    task.scheduler.load_state_dict(r['scheduler'])
    task.data_rng.set_state(r['data_rng'])
    task.target_rng.setstate(r['target_rng'])
    torch.set_rng_state(r['torch_rng'])
    task.continuation, task.step_index = q, step
