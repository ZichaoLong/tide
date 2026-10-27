"""Independent formula anchors for 0770593 affine-delta/GDN/SSD §1–5.

Scalar textbook examples, a dense grouped-convolution sequence oracle and
first-order directional differences are independent of the graph schedules.
"""
import math
from dataclasses import replace
import pytest
import torch
from tidegraph import Graph, Node, Region
from tidegraph.content import Content
from tidegraph.weighted import RecurrentConfig, RegionalConfig, bind


def local(dtype, family, **kwargs):
    c = RecurrentConfig(memory_family=family, **kwargs)
    b = bind(Graph((Node(0),), (), (Region(1),), (0,), (0,)), memories=c,
             regions=RegionalConfig(region_control_mode="fixed", region_evidence=None, total_fraction=.5), dtype=dtype)
    w = b.model.nodes[0]
    return w, w.kernel, w.full_program.content


def inverse_silu(y):
    lo, hi = 0., max(2., 2*y)
    for _ in range(80):
        x = (lo+hi)/2
        if x/(1+math.exp(-x)) < y: lo = x
        else: hi = x
    return (lo+hi)/2


@pytest.mark.parametrize("family", ["affine", "delta", "gated_delta", "ssd"])
def test_recurrent_textbook_scalar_examples(dtype, family):
    args = dict(width=1, summary_width=1, key_width=1, value_width=1, ffn_width=1,
                epsilon=1., alpha_ff=0., memory_read="state")
    if family in {"gated_delta", "ssd"}: args["output_epsilon"] = 1.
    if family == "gated_delta": args["qk_epsilon"] = 1.
    w, m, f = local(dtype, family, **args)
    with torch.no_grad():
        for p in m.parameters(): p.zero_()
        for p in f.parameters(): p.zero_()
        m.summary.fill_(math.sqrt(2))
        if family in {"affine", "delta"}:
            m.key.fill_(math.sqrt(2)); m.value.fill_(math.sqrt(2)); m.read_query.fill_(math.sqrt(2))
            m.read_out.fill_(1); f.query.fill_(math.sqrt(2)); f.out[1].fill_(1)
        else:
            m.conv.weight.fill_(1); m.read_out.fill_(1); f.out.fill_(1); f.rho.fill_(1)
            if family == "gated_delta":
                m.projection.copy_(torch.tensor([inverse_silu(1/math.sqrt(3))]*2+[inverse_silu(2)],dtype=dtype)[:,None]*math.sqrt(2))
                f.output_gate.fill_(math.sqrt(2))
            else:
                p = [1, inverse_silu(2), inverse_silu(1), inverse_silu(3), 0]
                m.projection.copy_(torch.tensor(p,dtype=dtype)[:,None]*math.sqrt(2))
                m.delta_bias.fill_(math.log(math.expm1(.5))); m.log_rate.fill_(math.log(2*math.log(2)))
                f.direct.fill_(.25)
    u = torch.ones(1,dtype=dtype)
    initial = m.initial(w)
    assert initial.slots["time"].dtype == torch.int64 and initial.slots["time"].item() == 0
    q = m.step(w,initial,Content(torch.tensor([1.,1.],dtype=dtype)),0)
    expected_h = {"affine":1., "delta":.5, "gated_delta":.5, "ssd":1.}[family]
    torch.testing.assert_close(q.slots["H"],torch.full_like(q.slots["H"],expected_h))
    for name in ("Z","W"): torch.testing.assert_close(q.slots[name],torch.ones_like(q.slots[name]))
    if family == "gated_delta": expected = 1+float(torch.nn.functional.silu(u))/math.sqrt(17)
    elif family == "ssd":
        y = 3.5*float(torch.nn.functional.silu(u)); expected = 1+y/math.sqrt(1+y*y)
    else: expected = 1+expected_h
    torch.testing.assert_close(f(m,q,0,u),u.new_tensor([expected]))
    if family in {"gated_delta","ssd"}:
        second = m.step(w,q,Content(torch.tensor([1.,1.],dtype=dtype)),1)
        torch.testing.assert_close(second.slots["H"],torch.full_like(second.slots["H"],23/32 if family=="gated_delta" else 1.5))
        # The same current convolution is read from old and proposal caches.
        a,b = m.features(initial,0,u),m.features(q,0,u)
        for x,y in zip(a,b): torch.testing.assert_close(x,y)


def normalized(x, eps):
    return x/torch.sqrt(eps+(x*x).mean(-1,keepdim=True))


def dense_reference(m, f, inputs):
    """Ordinary unit-receive sequence; grouped conv1d, literal mixer equations.

    Does not call candidate step/effective/features/matrix_read/Full or Conv.
    Returns every matrix, Mem state long vector, and final content.
    """
    c=m.config; n=len(inputs); x=normalized(inputs,c.epsilon)
    raw=x@m.projection.t()
    if c.memory_family=="ssd":
        output_gate,raw,delta=raw.split((c.heads*c.value_width,c.conv_width,c.heads),-1)
        delta=torch.nn.functional.softplus(delta+m.delta_bias)
    convolved=torch.nn.functional.conv1d(torch.nn.functional.pad(raw.t()[None],(c.conv_size-1,0)),
              m.conv.weight.flip(0).t()[:,None],m.conv.bias,groups=c.conv_width)[0].t()
    features=torch.nn.functional.silu(convolved)
    if c.memory_family=="gated_delta":
        q,k,v=features.reshape(n,c.heads,-1).split((c.key_width,c.key_width,c.value_width),-1)
        q=q/torch.sqrt(c.qk_epsilon+(q*q).sum(-1,keepdim=True))/math.sqrt(c.key_width)
        k=k/torch.sqrt(c.qk_epsilon+(k*k).sum(-1,keepdim=True))
        delta=torch.nn.functional.softplus(x@m.delta.t()+m.delta_bias)
        beta=torch.sigmoid(x@m.write.t()+m.write_bias)
    else:
        v,k,q=features.split((c.heads*c.value_width,c.groups*c.key_width,c.groups*c.key_width),-1)
        v=v.reshape(n,c.heads,c.value_width)
        k=k.reshape(n,c.groups,c.key_width).repeat_interleave(c.heads//c.groups,dim=1)
        q=q.reshape(n,c.groups,c.key_width).repeat_interleave(c.heads//c.groups,dim=1)
    retain=torch.exp(-m.log_rate.exp()*delta)
    H=inputs.new_zeros(c.matrix_shape); states=[]; outputs=[]; mem=[]
    for i in range(n):
        if c.memory_family=="gated_delta":
            A=retain[i,:,None,None]*(torch.eye(c.key_width,dtype=inputs.dtype)[None]-beta[i,:,None,None]*k[i,:,None,:]*k[i,:,:,None])
            H=H@A+beta[i,:,None,None]*v[i,:,:,None]*k[i,:,None,:]
        else:
            H=retain[i,:,None,None]*H+delta[i,:,None,None]*v[i,:,:,None]*k[i,:,None,:]
        states.append(H);read=(H@q[i,:,:,None]).squeeze(-1);mem.append(m.read_out@read.flatten())
        if c.memory_family=="gated_delta":
            rms=f.rho*read/torch.sqrt(c.output_epsilon+(read*read).mean(-1,keepdim=True))
            mixed=f.out@(rms.flatten()*torch.nn.functional.silu(f.output_gate@x[i]))
        else:
            y=(read+f.direct[:,None]*v[i]).flatten()*torch.nn.functional.silu(output_gate[i])
            mixed=f.out@(f.rho*y/torch.sqrt(c.output_epsilon+(y*y).mean()))
        z=inputs[i]+c.alpha_mem*mixed; nx=normalized(z,c.epsilon)
        outputs.append(z+c.alpha_ff*(f.down@(torch.nn.functional.silu(f.gate@nx)*(f.up@nx))))
    return torch.stack(states),torch.stack(mem),torch.stack(outputs)


@pytest.mark.parametrize("family", ["gated_delta","ssd"])
def test_full_dense_mixer_correspondence_and_all_parameter_vjp(dtype,family):
    w,m,f=local(dtype,family,width=3,summary_width=2,key_width=2,value_width=3,
                heads=4,groups=2 if family=="ssd" else 1,conv_size=3,conv_stride=7)
    inputs=torch.linspace(-.6,.8,15,dtype=dtype).reshape(5,3).requires_grad_()
    q=m.initial(w); states=[];mem=[];outputs=[]
    for i,u in enumerate(inputs):
        q=m.step(w,q,Content(torch.cat((u.new_ones(1),u))),7*i+2)
        states.append(q.slots["H"]);mem.append(m.mem(q,7*i+2,u)[1]);outputs.append(f(m,q,7*i+2,u))
    actual=tuple(torch.stack(v) for v in (states,mem,outputs));expected=dense_reference(m,f,inputs)
    for a,b in zip(actual,expected): torch.testing.assert_close(a,b)
    roots=(inputs,*m.parameters(),*f.parameters())
    a=torch.autograd.grad(sum(x.square().sum() for x in actual),roots,allow_unused=True,retain_graph=True)
    b=torch.autograd.grad(sum(x.square().sum() for x in expected),roots,allow_unused=True)
    for x,y in zip(a,b):
        if x is None or y is None: assert x is None and y is None
        else: torch.testing.assert_close(x,y,rtol=1e-4 if dtype==torch.float32 else 1e-8,atol=1e-7 if dtype==torch.float32 else 1e-10)


@pytest.mark.parametrize("family", ["affine","delta","gated_delta","ssd"])
def test_recurrent_weak_zero_and_original_time(dtype,family):
    w,m,f=local(dtype,family,**({} if family in {"affine","delta"} else dict(conv_size=3,conv_stride=2,matrix_rates=(.07,))))
    u=torch.linspace(-.2,.5,m.config.width,dtype=dtype)
    clock=2**54+3
    q=m.step(w,m.initial(w),Content(torch.cat((u.new_ones(1),u))),clock)
    assert q.slots["time"].item()==clock
    zero=m.step(w,q,Content(torch.cat((u.new_zeros(1),u*1.3))),clock+2)
    assert zero is q
    baseline=m.effective(q,clock+9)
    small=m.step(w,q,Content(torch.cat((u.new_tensor([1e-5]),u*1.3))),clock+2)
    for a,b in zip(baseline,m.effective(small,clock+9)):
        torch.testing.assert_close(a,b,atol=2e-5,rtol=2e-5)
    # Exact timestamps are different even when effective states converge.
    assert small.slots["time"].item()==clock+2 and q.slots["time"].item()==clock
    # Nonuniform times decay by elapsed time, not reception count.
    expected=torch.exp(u.new_tensor(-9*m.config.lambda_short))*q.slots["W"][0]
    torch.testing.assert_close(baseline[2][0],expected)
    m.validate(w,q);m.validate(w,small)
    invalid=replace(q,slots={**q.slots,"time":q.slots["time"].to(dtype)})
    with pytest.raises(ValueError,match="timestamp"): m.validate(w,invalid)


@pytest.mark.parametrize("family", ["affine","delta","gated_delta","ssd"])
def test_recurrent_local_directional_derivative(dtype,family):
    if dtype != torch.float64: return
    w,m,f=local(dtype,family,**({} if family in {"affine","delta"} else dict(conv_size=2,conv_stride=3)))
    parameters=tuple(m.parameters())+tuple(f.parameters())
    directions=tuple(torch.linspace(-.2,.3,p.numel(),dtype=dtype).reshape_as(p) for p in parameters)
    u=torch.linspace(-.3,.4,m.config.width,dtype=dtype)
    def value():
        q=m.initial(w)
        for time,mass in ((1,.6),(4,.8),(11,.3)):
            q=m.step(w,q,Content(torch.cat((u.new_tensor([mass]),u))),time)
        mem=m.mem(q,12,u)
        return f(m,q,12,u).square().sum()+sum(v.square().sum() for v in mem)+q.slots["Z"].square().sum()
    gradients=torch.autograd.grad(value(),parameters,allow_unused=True)
    grad=sum((g*d).sum() for g,d in zip(gradients,directions) if g is not None)
    original=tuple(p.detach().clone() for p in parameters)
    def perturb(epsilon):
        with torch.no_grad():
            for p,o,d in zip(parameters,original,directions):p.copy_(o+epsilon*d)
    perturb(1e-5);plus=value().detach()
    perturb(-1e-5);minus=value().detach()
    perturb(0)
    torch.testing.assert_close((plus-minus)/2e-5,grad,rtol=1e-5,atol=1e-8)


def test_recurrent_configuration_applicability():
    for args in (dict(memory_read="qry"),dict(heads=2),dict(conv_size=3),dict(matrix_rates=(.1,)),
                 dict(memory_family="ssd",heads=3,groups=2),dict(memory_family="gated_delta",groups=2),
                 dict(qk_epsilon=.1),dict(memory_family="ssd",conv_size=2,conv_stride=2**63-1)):
        with pytest.raises(ValueError):RecurrentConfig(**args)


@pytest.mark.parametrize("family",["affine","delta"])
@pytest.mark.parametrize("mode",["sum","state"])
def test_basic_matrix_equations_general_width_and_vjp(dtype,family,mode):
    w,m,f=local(dtype,family,width=3,summary_width=2,key_width=2,value_width=3,memory_read=mode)
    xs=torch.linspace(-.4,.7,12,dtype=dtype).reshape(4,3).requires_grad_()
    q=m.initial(w); H=q.slots["H"];Z=q.slots["Z"];W=q.slots["W"];last=0
    actual=[];expected=[]
    for time,receive,u in zip((0,2,7,8),(.6,.2,1.3,.7),xs):
        x=normalized(u,m.config.epsilon);weight=min(1.,receive)
        decay=u.new_tensor([math.exp(-m.config.lambda_short*(time-last)),1.]);last=time
        H=decay[:,None,None]*H;Z=decay[:,None]*Z;W=decay*W
        raw=m.key@x;k=raw/torch.maximum(raw.norm(),u.new_tensor(1.));v=m.value@x
        g=torch.exp(-weight*torch.logaddexp(m.retain@x+m.retain_bias,u.new_tensor(0.)))
        if family=="affine": A=g*torch.eye(m.config.key_width,dtype=dtype);B=weight*torch.outer(v,k)
        else:
            beta=weight/(1+torch.exp(-(m.write@x+m.write_bias)))
            A=g*(torch.eye(m.config.key_width,dtype=dtype)-beta*torch.outer(k,k));B=beta*torch.outer(v,k)
        H=H@A+B;Z=Z+weight*(m.summary@x);W=W+weight
        q=m.step(w,q,Content(torch.cat((u.new_tensor([receive]),u))),time)
        for name,value in (("H",H),("Z",Z),("W",W)):
            torch.testing.assert_close(q.slots[name],value);actual.append(q.slots[name]);expected.append(value)
        mem=Z/W.clamp_min(1)[:,None] if mode=="sum" else torch.stack([m.read_out[a]@H[a]@m.read_query@x for a in range(2)])
        got=m.mem(q,time,u)
        torch.testing.assert_close(torch.stack(got[:2]),mem)
        actual.extend(got[:2]);expected.extend(mem.unbind())
        mixed=sum(f.out[a]@H[a]@f.query@x for a in range(2));full=u+f.config.alpha_mem*mixed
        nx=normalized(full,f.config.epsilon)
        full=full+f.config.alpha_ff*(f.down@(torch.nn.functional.silu(f.gate@nx)*(f.up@nx)))
        actual.append(f(m,q,time,u));expected.append(full)
        torch.testing.assert_close(actual[-1],full)
    roots=(xs,*m.parameters(),*f.parameters())
    ga=torch.autograd.grad(sum(v.square().sum() for v in actual),roots,allow_unused=True,retain_graph=True)
    ge=torch.autograd.grad(sum(v.square().sum() for v in expected),roots,allow_unused=True)
    for a,b in zip(ga,ge):
        if a is None or b is None: assert a is None and b is None
        else: torch.testing.assert_close(a,b)


@pytest.mark.parametrize("family",["affine","delta","gated_delta","ssd"])
@pytest.mark.parametrize("mode",["0","-","+"])
def test_independent_read_snapshots_and_unselected_state_retention(dtype,family,mode):
    from tidegraph import Continuation,External
    from tidegraph.reference import run
    c=RecurrentConfig(memory_family=family,width=1,summary_width=1,key_width=1,value_width=1,
                       ffn_width=1,memory_read="sum",epsilon=1.,alpha_ff=0.)
    r=RegionalConfig(read_mode=mode,hidden_width=2,region_control_width=1,region_control_mode="fixed",
                     region_evidence=None,total_fraction=0.)
    b=bind(Graph((Node(0),Node(0)),(),(Region(2),),(0,1),(0,1)),memories=c,regions=r,dtype=dtype)
    with torch.no_grad():
        for p in b.model.parameters():
            if p.requires_grad:p.fill_(.3)
        for v,w in enumerate(b.model.nodes):
            w.kernel.summary.fill_(math.sqrt(2))
            score=w.read_program.score
            score.matrix.zero_();score.matrix[:,3].copy_(torch.tensor([1.,-1.],dtype=dtype))
            score.hidden_bias.zero_();score.vector.copy_(torch.tensor([1.,-1.],dtype=dtype));score.bias.fill_(-100*v)
    xs=[External(0,v,i,time,torch.tensor([1.,u],dtype=dtype)) for v in range(2) for i,(time,u) in enumerate(((0,1.),(2,2.)))]
    result=run(b.graph,b.model,Continuation(b.graph.identity,1),xs,3,sealed_until=3)
    p2=2*math.sqrt(2/5)
    expected={"0":0.,"-":1.,"+":(1+p2)/2}[mode]
    last=[row for row in result.trace if row["time"]==2]
    for v,row in enumerate(last):
        torch.testing.assert_close(row["descriptor"]["score"],torch.tensor(expected-100*v,dtype=dtype))
        torch.testing.assert_close(row["descriptor"]["w"],torch.tensor(1.,dtype=dtype))
        torch.testing.assert_close(row["descriptor"]["mu"],torch.tensor(1.,dtype=dtype))
        assert row["descriptor"]["evidence"].count_nonzero()==0
    assert result.continuation.states[0,1].observations==2
    torch.testing.assert_close(result.continuation.states[0,1].slots["Z"][1],torch.tensor([1+p2],dtype=dtype))
    assert all(row[2]==0 for row in result.outputs)  # node 1 updated despite c=0.
    # Full always reads proposal, independently of the descriptor snapshot.
    w=b.model.nodes[0];u=torch.tensor([2.],dtype=dtype)
    old=w.kernel.step(w,w.kernel.initial(w),Content(torch.tensor([1.,1.],dtype=dtype)),0)
    proposal=result.continuation.states[0,0]
    expected_full=w.full_program.content(w.kernel,proposal,2,u)
    torch.testing.assert_close(result.outputs[-1][-1][1:],expected_full)
    assert not torch.equal(expected_full,w.full_program.content(w.kernel,old,2,u))
