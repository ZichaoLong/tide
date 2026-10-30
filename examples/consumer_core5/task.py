"""Downstream-owned toy data/head/loss/SGD; public core-5 execution underneath."""
import random
import torch
from tidegraph import Continuation, ExecutionPlacement
from tidegraph.planning.recipes import prepare
from tidegraph.planning.materialize import materialize
from tidegraph.planning.facts import describe
from tidegraph.settle import SettleGraph
from tidegraph.weighted import layered_inputs, Payload
from tidegraph.weighted.execution import eager_runner
from recording import fingerprint, snapshot


class Task:
    def __init__(self, recipe, execution, seed, *, native_library=None, reference=False):
        if execution['requested']['device'] != 'cpu':
            raise ValueError('this starter checks CPU; accelerator clients have separate gates')
        self.execution, self.seed = execution, seed
        self.batch, self.positions = 2, 2
        self.plan, _ = prepare(recipe, samples=self.batch, positions=self.positions, seed=seed)
        if self.plan['graph']['kind'] != 'ranked-local':
            raise ValueError('starter requires a ports-bound Settle recipe')
        dtype = getattr(torch, execution['requested']['dtype'])
        self.binding, _, self.span = materialize(self.plan, max_bytes=16*1024*1024, dtype=dtype)
        self.spec = SettleGraph(self.binding.graph, tuple(self.plan['graph']['ranks']),
                                binding='ports', time_stride=self.plan['stride'])
        torch.manual_seed(seed)
        self.head = torch.nn.Linear(self.plan['width'], 1, dtype=dtype)
        self.data_rng = torch.Generator().manual_seed(seed+1)
        self.target_rng = random.Random(seed+2)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=.0001, momentum=.3, weight_decay=.01)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=.9)
        placement = ExecutionPlacement(scoring_dtype=execution['requested']['scoring_dtype'])
        if reference:
            from tidegraph.reference import run
            self.execute = lambda *a, **kw: run(self.binding.graph, self.binding.model, *a, **kw)
        else:
            e = execution['requested']
            self.execute = eager_runner(self.binding, 'native' if e['client'] == 'python-native' else 'python',
                placement, native_library, streaming=e['schedule'] == 'streaming', workers=e['workers'], packing=e['packing'])
        self.continuation = Continuation(self.binding.graph.identity, self.batch)
        self.step_index = 0
        # planning.materialize's synthetic inputs are unused. This application
        # has its own data policy and workload identity, not the benchmark's ID.
        self.workload = dict(schema='core5-consumer-task-v1', model_id=self.plan['model_id'], seed=seed,
            batch=self.batch, positions=self.positions, input='cpu-generator normal*.2',
            target='python-generator uniform(-.1,.1)', head='linear(width,1), dropout=.1',
            loss='mean squared error of numeric cG per present output',
            optimizer='SGD lr=.0001 momentum=.3 weight_decay=.01', scheduler='StepLR step=1 gamma=.9',
            gradient_boundary='detach graph state after each backward before optimizer')
        self.workload_id = fingerprint(self.workload)
        self.application_execution = dict(schema='example-core5-execution-v1',
            foundation_execution_id=execution['execution_id'], autograd_multithreading=True)

    def parameters(self):
        return [p for p in self.binding.model.parameters() if p.requires_grad]+list(self.head.parameters())

    def named_parameters(self):
        return dict([('graph.'+n, p) for n, p in self.binding.model.named_parameters()]+[
                    ('head.'+n, p) for n, p in self.head.named_parameters()])

    def forward(self, x, *, training):
        start = self.step_index*self.positions
        inputs = layered_inputs(self.spec, x, start_position=start)
        stop = (self.step_index+1)*self.span
        result = self.execute(self.continuation, inputs, stop, sealed_until=stop)
        if not result.outputs:
            raise RuntimeError('toy task requires at least one actual output')
        # Payload contains (c,G); numerical projection multiplies by c once.
        features = torch.stack([Payload.decode(v).numerical()[1] for *_, v in result.outputs])
        predictions = self.head(torch.nn.functional.dropout(features, p=.1, training=training))
        return result, predictions

    def step(self):
        self.optimizer.zero_grad(set_to_none=True)
        x = (.2*torch.randn(self.batch, self.positions, self.plan['width'], generator=self.data_rng,
                           dtype=self.head.weight.dtype)).requires_grad_()
        result, predictions = self.forward(x, training=True)
        target = self.target_rng.uniform(-.1, .1)
        loss = (predictions-target).square().mean()
        # Record the actual engine policy separately from native forward workers.
        with torch.autograd.set_multithreading_enabled(True):
            loss.backward()
        if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.parameters()):
            raise FloatingPointError('nonfinite loss or gradient')
        self.continuation = result.continuation.detach()
        self.optimizer.step()
        self.scheduler.step()
        self.step_index += 1
        return snapshot(dict(result=result, loss=loss, input=x, input_gradient=x.grad,
                             predictions=predictions, target=target, training=self.state()))

    def state(self):
        return snapshot(dict(parameters=self.named_parameters(),
            gradients={n:p.grad for n,p in self.named_parameters().items()}, optimizer=self.optimizer.state_dict(),
            scheduler=self.scheduler.state_dict(), continuation=self.continuation,
            step=self.step_index, position=self.step_index*self.positions,
            data_rng=self.data_rng.get_state(), target_rng=self.target_rng.getstate(), torch_rng=torch.get_rng_state()))

    def facts(self):
        return dict(configuration=describe(self.plan), workload=self.workload, workload_id=self.workload_id,
                    materialization_plan_id=self.plan['sha256'], execution=self.execution,
                    application_execution=self.application_execution,
                    application_execution_id=fingerprint(self.application_execution))

    def infer(self):
        with torch.no_grad():
            x = torch.zeros(self.batch, self.positions, self.plan['width'], dtype=self.head.weight.dtype)
            _, predictions = self.forward(x, training=False)
        return snapshot(predictions)
