"""Tiny independent MoE mechanism checks, separate from language learning."""
import copy
import torch
from .moe import Top2Experts,MoETransformer,objective_backward
from .language_checks import finite_delta


def mechanism_checks(device):
    torch.manual_seed(727)
    reference=Top2Experts(16,32,20,2,'reference')
    model=copy.deepcopy(reference).to(device);model.implementation='bucketed_bmm'
    x=torch.randn(2,5,16,requires_grad=True);xn=x.detach().to(device).requires_grad_()
    mask=torch.tensor([[1,1,1,1,1],[1,1,0,0,0]],dtype=torch.bool)
    reference.valid_mask=mask;model.valid_mask=mask.to(device)
    y=reference(x);yn=model(xn);v=torch.randn_like(y)
    def aux(m):
        s=m.stats;n=s['valid']
        return 20*((s['counts']/(2*n))*(s['p_sum']/n)).sum()+.1*s['z_sum']/n
    ((y*v).sum()+aux(reference)).backward();((yn*v.to(device)).sum()+aux(model)).backward()
    checks=dict(forward=finite_delta(yn,y,1e-6,1e-5),input_vjp=finite_delta(xn.grad,x.grad,1e-6,1e-5),
        parameters={n:finite_delta(p.grad,dict(reference.named_parameters())[n].grad,1e-6,1e-5) for n,p in model.named_parameters()})
    s=model.stats
    assertions=dict(top2_indices_match_cpu=torch.equal(s['indices'].cpu(),reference.stats['indices']),
        top2_distinct=bool((s['indices'][:,0]!=s['indices'][:,1]).all()),
        valid_assignment_count=int(s['counts'].sum())==14,
        weights_sum_one=bool(torch.allclose(s['weights'].sum(-1),torch.ones(10,device=device),atol=1e-6,rtol=1e-5)),
        dropless_rows=20<=model.physical_rows<40,cold_experts_exercised=bool((s['counts']==0).any()),
        gradients_finite=all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()))
    if not all(assertions.values()):raise AssertionError('MoE routing mechanism failure: '+str(assertions))
    return dict(state='passed',discrete=assertions,strict_numerical=checks,
        scope='B2 L5 d16 h32 E20, CPU scalar-expert reference; input/router/expert+bias VJP includes aux, padding and empty experts')
