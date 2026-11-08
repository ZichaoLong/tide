"""Small independent numerical, discrete and mechanism axes for new LM paths."""
import copy
import torch
from torch.nn import functional as F
from .language import SequenceCore5,sequence_readout,IntegratedMLP
from .language_training import batch_from,prefix_cache,optimizer_for,update
from .train import cpu_tree


def finite_delta(a,b,atol,rtol):
    a=a.detach().double().cpu();b=b.detach().double().cpu()
    if a.shape!=b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise AssertionError('shape or nonfinite independent comparison')
    error=(a-b).abs();bad=error>atol+rtol*b.abs()
    return dict(state='failed' if bool(bad.any()) else 'passed',atol=atol,rtol=rtol,
        max_abs=float(error.max()),failed_elements=int(bad.sum()),elements=a.numel(),
        relative_l2=float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b).clamp_min(1e-30)))


def core_axes(encoder,options,seed,device):
    from scripts.check_core5 import canonical_result
    from scripts.assess_core5_axes import Audit,positive_support,root_alias_check
    from types import SimpleNamespace
    # Prior task/update probes must not contaminate the independent VJP.
    encoder.zero_grad(set_to_none=True)
    reference=SequenceCore5(seed=seed,**options)
    reference.load_state_dict(cpu_tree(encoder.state_dict()),strict=True)
    generator=torch.Generator().manual_seed(991)
    raw=torch.randn(2,3,encoder.width,generator=generator)*.2
    x=raw.detach().clone().to(device).requires_grad_(True)
    xr=raw.detach().clone().requires_grad_(True)
    result=encoder.run_features(x,trace=True)
    expected=reference.run_features(xr,implementation='reference',trace=True)
    actual_tree=canonical_result(result);reference_tree=canonical_result(expected)
    audit=Audit();audit.visit(actual_tree,reference_tree,'complete_execution')
    support=positive_support(actual_tree);expected_support=positive_support(reference_tree)
    if support!=expected_support:raise AssertionError('reference/eager discrete support mismatch')
    roots=root_alias_check(SimpleNamespace(encoder=encoder),x)
    y=sequence_readout(encoder,result,x);yr=sequence_readout(reference,expected,xr)
    weights=torch.linspace(-.1,.2,y.numel()).reshape(y.shape)
    (y*weights.to(device)).sum().backward();(yr*weights).sum().backward()
    audit.visit(cpu_tree(y),cpu_tree(yr),'all_position_readout')
    audit.visit(cpu_tree(x.grad),cpu_tree(xr.grad),'all_position_input_vjp')
    gradients={n:None if p.grad is None else cpu_tree(p.grad) for n,p in encoder.named_parameters()}
    ref_gradients={n:None if p.grad is None else cpu_tree(p.grad) for n,p in reference.named_parameters()}
    audit.visit(gradients,ref_gradients,'shared_parameters_vjp')
    report=audit.report()
    if report['nonfinite'] or report['discrete']['mismatches']:
        raise AssertionError('independent execution has nonfinite/discrete failure: '+str(report))
    if x.grad is None or not bool(x.grad.abs().sum()>0):raise AssertionError('input gradient disconnected')
    if not any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in encoder.parameters()):
        raise AssertionError('graph gradient disconnected')
    causal_input=x.detach().clone().requires_grad_(True)
    causal_output=encoder(causal_input)
    future_vjp=torch.autograd.grad(causal_output[:,:2].sum(),causal_input)[0][:,2:]
    if bool(future_vjp.ne(0).any()):raise AssertionError('future input has a nonzero VJP to earlier output')
    with torch.no_grad():
        changed=x.detach().clone();changed[:,2:]+=3
        original=encoder(x.detach())
        changed_result=encoder.run_features(changed,trace=True)
        perturbed=sequence_readout(encoder,changed_result,changed)
        causal=finite_delta(perturbed[:,:2],original[:,:2],0,0)
        changed_cpu=changed.cpu()
        changed_reference=reference.run_features(changed_cpu,implementation='reference',trace=True)
        reference_perturbed=sequence_readout(reference,changed_reference,changed_cpu)
        reference_causal=finite_delta(reference_perturbed[:,:2],yr.detach()[:,:2],0,0)
        cutoff=2*encoder.spec.stride
        early_support=[r for r in support if r[0]<cutoff]
        changed_support=[r for r in positive_support(canonical_result(changed_result)) if r[0]<cutoff]
        if reference_causal['state']!='passed' or early_support!=changed_support:
            raise AssertionError('reference causality or early discrete support changed')
        chunked=encoder(x.detach(),chunk_size=1)
        streaming=encoder(x.detach(),implementation='streaming')
    return dict(strict=report,roots=roots,causal=causal,
        causal_mechanisms=dict(future_vjp_exact_zero=True,reference_intervention=reference_causal,
            early_support_unchanged=True,cutoff=cutoff,
            note='retain strict eager perturbation failure separately; independent reference, exact future VJP and early support checked; dynamic packing is a hypothesis, not a proven root cause'),
        chunked=finite_delta(chunked,original,1e-6,1e-5),streaming=finite_delta(streaming,original,1e-6,1e-5),
        input_gradient_by_position=cpu_tree(x.grad.norm(dim=-1)).tolist(),
        numerical_note='finite strict disagreements retained after complete records, VJP, support, roots and causality checks; cause not assumed roundoff')


def ce_formula(logits,labels):
    values=logits.detach().clone().requires_grad_(True)
    loss=F.cross_entropy(values.flatten(0,1),labels.flatten(),ignore_index=-100)
    grad=torch.autograd.grad(loss,values)[0]
    expected=values.detach().softmax(-1)
    valid=labels!=-100;safe=labels.clamp_min(0)
    expected.scatter_add_(-1,safe.unsqueeze(-1),-torch.ones_like(safe,dtype=values.dtype).unsqueeze(-1))
    expected*=valid.unsqueeze(-1);expected/=valid.sum()
    return finite_delta(grad,expected,1e-6,1e-5)


def a_checks(model,rows,plan,kind,seed,device,depth=1):
    b=batch_from(rows,torch.arange(2),device);x=b['input_ids'][:,:3];labels=b['labels'][:,:3]
    model.eval();logits=model(x)
    loss=F.cross_entropy(logits.flatten(0,1),labels.flatten())
    loss.backward()
    if model.embedding.weight is not model.head.weight:raise AssertionError('tied task head lost identity')
    if not bool(model.embedding.weight.grad.abs().sum()>0):raise AssertionError('embedding/head disconnected')
    optimizer=optimizer_for(model,plan,'a',kind)
    ids=[id(p) for g in optimizer.param_groups for p in g['params']]
    if len(ids)!=len(set(ids)):raise AssertionError('duplicate optimizer parameter')
    before=model.embedding.weight.detach().clone();optimizer.step()
    if torch.equal(before,model.embedding.weight):raise AssertionError('task optimizer failed to update')
    checks=dict(task_loss_finite=True,tied_head=True,optimizer_unique=True,optimizer_updates=True,
                analytic_ce_vjp=ce_formula(logits,labels))
    if kind.startswith('tide'):
        from .language_training import core_options
        checks['core']=core_axes(model.encoder,core_options(plan,kind,depth),seed,device)
    return dict(state='completed',axes=dict(execution='passed',mechanisms='passed',
        strict_numerical='inspect detailed original thresholds',discrete_semantics='passed',
        recovery='unchecked_here',learning='unchecked',performance='unchecked'),checks=checks,
        admitted_for_finite_exploration=checks['analytic_ce_vjp']['state']=='passed')


def b_checks(model,lm,token_rows,plan,kind,device,seed=11,depth=1):
    rows={k:v[:2,:3] if isinstance(v,torch.Tensor) else v[:2] for k,v in token_rows.items()}
    features=prefix_cache(lm,rows,device,2)
    b=batch_from(features,torch.arange(2),device)
    inputs=dict(input_ids=b['input_ids'],attention_mask=(b['labels']!=-100).long(),use_cache=False)
    with torch.no_grad():
        full=lm(**inputs,return_dict=True).logits
        baseline=model.head(model.norm(b['residual']+b['teacher']))
    tolerance=plan['checks'];checks=dict(frozen_prefix=finite_delta(baseline,full,tolerance['qwen_atol'],tolerance['qwen_rtol']))
    optimizer=optimizer_for(model,plan,'b',kind)
    entered=[]
    for step in range(2):
        update(model,optimizer,b,plan,'b',kind,step,20,device)
        entered.append([n for n,p in model.selected.named_parameters() if p.grad is not None and bool(p.grad.abs().sum()>0)])
    if not entered[-1]:raise AssertionError('selected module gradient disconnected')
    if kind.startswith('tide') and not any(n.startswith('core.') for n in entered[-1]):
        raise AssertionError('TIDE gradient did not enter after output warm start')
    original=lm.model.layers[-1].mlp
    try:
        with torch.no_grad():current_original=lm(**inputs,return_dict=True).logits
        neutral=IntegratedMLP(original,model.selected,alpha=0)
        lm.model.layers[-1].mlp=neutral
        with torch.no_grad():same=lm(**inputs,return_dict=True).logits
        checks['alpha_zero']=finite_delta(same,current_original,0,0)
        if checks['alpha_zero']['state']!='passed':raise AssertionError('alpha0 behavior preservation failed')
        integrated=IntegratedMLP(original,model.selected,alpha=1)
        lm.model.layers[-1].mlp=integrated
        with torch.no_grad():
            changed=lm(**inputs,return_dict=True).logits
            tail=model.logits(b,alpha=1)
            checks['full_replacement_tail']=finite_delta(changed,tail,tolerance['qwen_atol'],tolerance['qwen_rtol'])
        if integrated.original_calls:raise AssertionError('original MLP executed at alpha1')
        checks['original_mlp_calls_at_alpha1']=0
    finally:lm.model.layers[-1].mlp=original
    if kind.startswith('tide'):
        from .language_training import core_options
        checks['core']=core_axes(model.selected.core,core_options(plan,kind,depth),seed,device)
    checks.update(gradient_entry=entered,optimizer_unique=True,initial_local_output='train-only teacher mean before updates')
    return dict(state='completed',axes=dict(execution='passed',strict_numerical='inspect original tolerance results',
        discrete_semantics='passed',mechanisms='passed',recovery='unchecked_here',learning='unchecked'),
        checks=checks,admitted_for_finite_exploration=True)
