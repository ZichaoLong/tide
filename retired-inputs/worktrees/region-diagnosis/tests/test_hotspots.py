"""Packing candidates preserve the selected mathematical function and VJP."""
import torch

from benchmarks.hotspots import candidate
from tide import ModelConfig,TideModel


def test_bucketed_full_keeps_values_shared_parameter_and_input_gradients():
    torch.manual_seed(491)
    model=TideModel(ModelConfig(d=64,sharing="region_shared",attention_chunk=17)).double()
    x=torch.randn(2,19,64,dtype=torch.float64,requires_grad=True)
    reference=model(x)
    expected=torch.autograd.grad(reference.readout().square().sum(),(x,*model.parameters()),allow_unused=True)
    with candidate(model,"bucketed"):
        result=model(x)
        actual=torch.autograd.grad(result.readout().square().sum(),(x,*model.parameters()),allow_unused=True)
    torch.testing.assert_close(result.values,reference.values,atol=2e-12,rtol=2e-12)
    assert torch.equal(result.strengths>0,reference.strengths>0)
    for got,want in zip(actual,expected):
        assert (got is None)==(want is None)
        if got is not None:
            torch.testing.assert_close(got,want,atol=1e-9,rtol=2e-10)
