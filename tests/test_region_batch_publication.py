"""Region §4/6: stable ID uploads and support publication preserve private roots."""
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from tidegraph import History, Region
from tidegraph.compare import equivalent
from tidegraph.region import RegionInput, RegionLayout
from tidegraph.weighted.config import RegionalConfig
from tidegraph.weighted.programs import RegionalTotal
from tidegraph.weighted.region_batch import evaluate as numeric
from tidegraph.weighted.region_vjp import evaluate as differentiated
from isolated_cases import vjp
from test_weighted_region_batch import native_region


class MetadataCalls(TorchDispatchMode):
    """Count actual ATen literal construction/read calls, not helper counters."""
    def __init__(self):
        super().__init__()
        self.integers = 0
        self.scalar_reads = 0
        self.select_transposes = []

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if (function == torch.ops.aten.empty.memory_format and kwargs.get("dtype") == torch.int64
                and len(args[0]) > 0):
            self.integers += 1  # native at::tensor allocates its literal here
        elif (function == torch.ops.aten.lift_fresh.default and args[0].dtype == torch.int64
                and args[0].ndim > 0):
            self.integers += 1  # Python torch.tensor publishes its literal here
        if function == torch.ops.aten._local_scalar_dense.default:
            self.scalar_reads += 1
        if function == torch.ops.aten.select_backward.default:
            self.select_transposes.append(tuple(args[1]))
        return function(*args, **kwargs)


def fixture(dtype, mode, count):
    regional = mode in {"regional_total", "regional_total_zero"}
    options = (dict(region_control_mode="fixed", total_fraction=0. if mode.endswith("zero") else .5)
               if regional else {})
    config = RegionalConfig(selection_mode=mode, k=1, **options)
    program = RegionalTotal("region-publication-test-v1", config, torch.Generator().manual_seed(19), dtype)
    # These IDs cannot make a round trip through FP64. Tie order must follow
    # the exact node identity, independently of the candidate traversal order.
    nodes = tuple(2**54+i for i in (9, 3, 17))
    layout = RegionLayout.create(Region(config.capacity(3), selector=program.profile), nodes)
    requests, roots = [], {}
    for row in range(count):
        fields = []
        for i, node in enumerate(nodes):
            values = dict(w=(.3, .4, .2)[i], mu=(.4, .7, .5)[i], score=(.2, .2, -1.)[i])
            descriptor = {name: torch.tensor(value, dtype=dtype, requires_grad=True) for name, value in values.items()}
            if regional:
                descriptor["evidence"] = torch.zeros(2, dtype=dtype, requires_grad=True)
            roots.update({f"{row}.{node}.{name}": value for name, value in descriptor.items()})
            fields.append((node, descriptor))
        requests.append(RegionInput(History(), 2**54+21+row, tuple(fields), layout, dtype, torch.device("cpu")))
    return program, requests, roots


@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
@pytest.mark.parametrize("mode", ["regional_total", "proportional", "regional_total_zero", "hard_selection"])
def test_node_identity_construction_is_bounded_by_groups(dtype, native, mode):
    sorted_mode = mode in {"regional_total_zero", "hard_selection"}
    for count in (2, 17):
        program, requests, roots = fixture(dtype, mode, count)
        with MetadataCalls() as calls:
            result = native_region(program, requests) if native else differentiated(program, requests)
        # Only sorted sending needs IDs, uploaded once for the complete group.
        assert calls.integers == int(sorted_mode)
        expected = [program.step(request) for request in requests]
        equivalent(result, expected)
        if sorted_mode:
            wanted = (.2, .7, 0.) if mode.endswith("zero") else (0., .7, 0.)
            for row in result:
                equivalent(torch.stack([row.controls[node] for node, _ in requests[0].candidates]),
                           torch.tensor(wanted, dtype=dtype))
        node = requests[0].candidates[1][0]
        equivalent(vjp(result[0].controls[node], roots), vjp(expected[0].controls[node], roots))
        equivalent(vjp(result[0].controls[node], roots, zero=True),
                   vjp(expected[0].controls[node], roots, zero=True))


def test_python_vjp_publication_reuses_numeric_support():
    for count in (2, 17):
        program, requests, roots = fixture(torch.float64, "proportional", count)
        with MetadataCalls() as direct_calls:
            expected = numeric(program, requests)
        with MetadataCalls() as wrapped_calls:
            actual = differentiated(program, requests)
        # The VJP wrapper must not add one host predicate per B×L×candidate.
        assert wrapped_calls.scalar_reads == direct_calls.scalar_reads
        equivalent(actual, expected)
        oracle = [program.step(request) for request in requests]
        node = requests[0].candidates[0][0]
        equivalent(vjp(actual[0].controls[node], roots), vjp(oracle[0].controls[node], roots))


@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_allocation_backward_assembles_complete_fields_once(dtype, native):
    for count in (2, 17):
        program, requests, roots = fixture(dtype, "regional_total", count)
        actual = native_region(program, requests) if native else differentiated(program, requests)
        expected = [program.step(request) for request in requests]
        def loss(results):
            return sum((i+1)*sum(value.square() for value in row.controls.values())
                       for i, row in enumerate(results))
        with MetadataCalls() as calls:
            gradients = vjp(loss(actual), roots)
        equivalent(gradients, vjp(loss(expected), roots))
        # A row SelectBackward allocates/zeros the entire source field. Unbind
        # returns every row through one transpose instead, including None bars.
        assert (count, 3) not in calls.select_transposes
        assert (count,) not in calls.select_transposes
