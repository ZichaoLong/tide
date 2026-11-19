#!/usr/bin/env python3
"""Finite witnesses for encoded descriptions and regional self-attention.

Standard-library reference arithmetic; not a neural runtime or training test.
The scalar worked example is checked against separately derived equations.
"""
import math
from itertools import permutations

from role_bounded_allocation_checks import allocate, bounds, close


def silu(x):
    return x/(1+math.exp(-x))


def norm(x, epsilon=1e-5):
    scale = math.sqrt(epsilon+sum(a*a for a in x)/len(x))
    return [a/scale for a in x]


def mv(matrix, x):
    return [sum(a*b for a,b in zip(row,x)) for row in matrix]


def add(x, y):
    return [a+b for a,b in zip(x,y)]


def features(u, receipt, p, short, long, wshort, wlong, epsilon):
    return (norm(u,epsilon)+[math.log1p(max(0,receipt))]+short+long+
            [math.log1p(wshort),math.log1p(wlong)]+
            [a*b for a,b in zip(p,short)]+[a*b for a,b in zip(p,long)])


def encode(f, matrix, bias, readout, score_bias=0, epsilon=1e-5):
    z = norm([silu(x) for x in add(mv(matrix,f),bias)],epsilon)
    return z, score_bias+sum(a*b for a,b in zip(readout,z))


def score(zs, local, weights, params, heads=1, epsilon=1e-5):
    size=len(params['Q'])
    assert size%heads==0
    dh=size//heads
    positive=[i for i,w in enumerate(weights) if w>0]
    result=list(local)
    if not positive:
        return result
    q,k,v=([mv(params[name],z) for z in zs] for name in ('Q','K','V'))
    for i in positive:
        context=[]
        for h in range(heads):
            start,end=h*dh,(h+1)*dh
            logits=[sum(a*b for a,b in zip(q[i][start:end],k[j][start:end]))/
                    math.sqrt(dh)+math.log(weights[j]) for j in positive]
            peak=max(logits)
            exps=[math.exp(x-peak) for x in logits]
            probs=[x/sum(exps) for x in exps]
            context += [sum(p*v[j][a] for p,j in zip(probs,positive))
                        for a in range(start,end)]
        y=add(zs[i],mv(params['O'],context))
        hidden=[silu(x) for x in add(mv(params['F1'],norm(y,epsilon)),params['b1'])]
        final=add(y,add(mv(params['F2'],hidden),params['b2']))
        result[i] += sum(a*(x-z) for a,x,z in zip(params['readout'],final,zs[i]))
    return result


def main():
    unit={name:[[1.0]] for name in ('Q','K','V','O','F1','F2')}
    unit.update(b1=[0.0],b2=[0.0],readout=[1.0])
    # f=[u_norm,log(1+r),Mshort,Mlong,logWshort,logWlong,p*Mshort,p*Mlong].
    def description(short):
        f=features([0],1,[0],[short],[0],0,0,1)
        assert len(f)==8
        return encode(f,[[0,0,1,0,0,0,0,0]],[0],[1],epsilon=1)
    ds=[description(0),description(1)]
    zs,local=map(list,zip(*ds))
    scores=score(zs,local,[1,1],unit,epsilon=1)
    q=math.tanh(.5)/2+.5  # sigmoid(1), computed independently of silu
    q=q/math.sqrt(1+q*q)
    p=1/(1+math.exp(-q*q))
    ya,yb=q/2,q*(1+p)
    expected=[y+(y/math.sqrt(1+y*y))/(1+math.exp(-y/math.sqrt(1+y*y)))
              for y in (ya,yb)]
    close(scores,expected)
    c=allocate(scores,[0,0],[1,1],1,1,1)
    close(c,[(1+expected[0]-expected[1])/2,(1-expected[0]+expected[1])/2])
    assert round(c[0],4)==.0331
    baseline=score([[0],[0]],[0,0],[1,1],unit,epsilon=1)
    close(allocate(baseline,[0,0],[1,1],1,1,1),[.5,.5])
    assert scores[0]>baseline[0]  # only the other node's memory changed

    # Unequal receipt, two heads, nonzero FFN: permutation and zero neutrality.
    eye=[[float(i==j) for j in range(4)] for i in range(4)]
    params={name:eye for name in ('Q','K','V','O','F1','F2')}
    params.update(b1=[.1,-.2,.3,-.4],b2=[.2,.1,0,-.1],readout=[1,-.5,.2,.7])
    zs=[[.1,.2,.3,.4],[-.4,.5,-.6,.7],[.8,-.9,.2,-.1]]
    local,weights=[.3,-.2,.1],[.2,.5,1]
    original=score(zs,local,weights,params,heads=2)
    for order in permutations(range(3)):
        actual=score([zs[i] for i in order],[local[i] for i in order],
                     [weights[i] for i in order],params,heads=2)
        close(actual,[original[i] for i in order])
    close(score(zs+[[1,1,1,1]],local+[999],weights+[0],params,heads=2),
          original+[999])
    close(score(zs,local,[0,0,0],params,heads=2),local)
    no_relation=dict(params,readout=[0]*4)
    close(score(zs,local,weights,no_relation,heads=2),local)

    # Q=K=0: each head reads the independently computed receipt mean.
    uniform=dict(params,Q=[[0]*4 for _ in range(4)],K=[[0]*4 for _ in range(4)])
    actual=score(zs,local,weights,uniform,heads=2)
    mean=[sum(w*z[a] for w,z in zip(weights,zs))/sum(weights) for a in range(4)]
    expected=[]
    for z,s in zip(zs,local):
        y=add(z,mean)
        correction=add(mean,add([silu(x) for x in add(norm(y),params['b1'])],params['b2']))
        expected.append(s+sum(a*b for a,b in zip(params['readout'],correction)))
    close(actual,expected)

    # Oscillating bounded weak descriptions vanish from other candidates'
    # scores; their own scores need not converge. Test complete allocation too.
    previous=math.inf
    for i,eps in enumerate((1e-2,1e-4,1e-6,1e-8)):
        weak=[(-1)**i]*4
        s=score(zs+[weak],local+[100],weights+[eps],params,heads=2)
        error=max(abs(a-b) for a,b in zip(original,s[:3]))
        assert error<previous
        previous=error
        mu=[min(1,2*w) for w in weights+[eps]]
        L,U=bounds(mu,[0]*4,mu,2,received=weights+[eps])
        for reg in ('quad','shannon'):
            c=allocate(s,[0]*4,mu,L,U,1,reg)
            assert 0<=c[-1]<=2*eps
    assert previous<1e-6
    print('PASS: encoded memory→cross-candidate scores→allocation; scalar example; '
          'two heads, weighted mean, permutations, zero/weak candidates, ind recovery')


if __name__=='__main__':
    main()
