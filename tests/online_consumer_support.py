"""Independent tensor/discrete comparisons for complete-consumer records."""
import torch
from tools.online_bench.records import observer

def same(actual, expected, *, atol=1e-6, rtol=1e-5, tensor_norm=False):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        if set(expected) == {"shape", "values"}:
            assert actual["shape"] == expected["shape"]
            reference = torch.tensor(expected["values"],dtype=torch.float64)
            # Explicit low-precision cross-dtype policy: infinity-norm relative
            # error remains meaningful for a small component after cancellation.
            scale = reference.abs().max().item() if reference.numel() else 0
            torch.testing.assert_close(torch.tensor(actual["values"],dtype=torch.float64),reference,
                                       atol=atol+rtol*scale if tensor_norm else atol,
                                       rtol=0 if tensor_norm else rtol)
        else:
            for k in expected:
                same(actual[k], expected[k], atol=atol, rtol=rtol, tensor_norm=tensor_norm)
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a,b in zip(actual,expected):
            same(a,b,atol=atol,rtol=rtol,tensor_norm=tensor_norm)
    else:
        assert actual == expected
