"""Independent value/manual VJP for the experimental pointwise control."""
import math
import torch


def reference(module,x,upstream):
    """CPU FP64 formula and explicit chain rule; no autograd/native GELU."""
    w0,w1,w2,w3=[p.detach().cpu().double() for p in
                 (module.input.weight,module.core[0].weight,module.core[2].weight,module.output.weight)]
    x=x.detach().cpu().double();u=upstream.detach().cpu().double()
    xm,xs,ym,ys=[getattr(module,k).detach().cpu().double() for k in ('x_mean','x_scale','y_mean','y_scale')]
    z=(x-xm)/xs;h=torch.tanh(z@w0.T);v=h@w1.T
    cdf=.5*(1+torch.erf(v/math.sqrt(2)));g=v*cdf;r=g@w2.T
    output=(r@w3.T)*ys+ym
    d=u*ys;dw3=d.T@r;dr=d@w3;dw2=dr.T@g;dg=dr@w2
    dv=dg*(cdf+v*torch.exp(-v*v/2)/math.sqrt(2*math.pi))
    dw1=dv.T@h;dh=dv@w1;dz=dh*(1-h*h);dw0=dz.T@z;dx=(dz@w0)/xs
    return output,dict(input=dx,**{'input.weight':dw0,'core.0.weight':dw1,
                                'core.2.weight':dw2,'output.weight':dw3})


def check(module,device,atol=1e-6,rtol=1e-5):
    from .language_checks import finite_delta
    dimension=module.input.in_features
    x=torch.linspace(-.7,.6,2*dimension).reshape(2,dimension).to(device).requires_grad_(True)
    upstream=torch.linspace(.3,-.2,2*dimension).reshape_as(x).to(device)
    value=module(x);params=dict(module.named_parameters())
    grads=torch.autograd.grad((value*upstream).sum(),[x,*params.values()])
    output,expected=reference(module,x,upstream)
    values=finite_delta(value.detach().cpu().double(),output,atol,rtol)
    comparisons={key:finite_delta(actual.detach().cpu().double(),expected[key],atol,rtol)
                 for key,actual in zip(['input',*params],grads)}
    if any(not torch.isfinite(g).all() or not bool(g.abs().sum()>0) for g in grads):
        raise AssertionError('pointwise input/parameter gradient is nonfinite or disconnected after warm start')
    causal_input=x.detach().reshape(1,2,dimension).requires_grad_(True)
    future=torch.autograd.grad(module(causal_input)[:,:1].sum(),causal_input)[0][:,1:]
    if torch.count_nonzero(future):raise AssertionError('pointwise control has future dependence')
    return dict(value=values,manual_vjp=comparisons,future_vjp_zero=True,
                scope='two rows; independent CPU FP64 formula and manual input/all-weight VJP')
