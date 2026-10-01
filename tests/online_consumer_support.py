"""Independent tensor/discrete comparisons for complete-consumer records."""
import torch
from tools.online_bench.records import observer

def same(actual, expected, *, atol=1e-6, rtol=1e-5):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        if set(expected) == {"shape", "values"}:
            assert actual["shape"] == expected["shape"]
            torch.testing.assert_close(torch.tensor(actual["values"],dtype=torch.float64),
                                       torch.tensor(expected["values"],dtype=torch.float64),atol=atol,rtol=rtol)
        else:
            for k in expected:
                same(actual[k], expected[k], atol=atol, rtol=rtol)
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a,b in zip(actual,expected):
            same(a,b,atol=atol,rtol=rtol)
    else:
        assert actual == expected
