"""Five independently written local memories on a fixed ranked time grid.

This is an experiment-specific sequence function, not a continuation or graph
execution API. Mathematical source: the pinned KV/affine/delta/GDN/SSD notes.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .direct_semantic_ops import (NodeLinear,norm,weighted_softmax,summary,time_weights,
                                 causal_conv,dual_recurrence)


def heads(x,h):
    n,b,l,d=x.shape
    return x.reshape(n,b,l,h,d//h).permute(0,1,3,2,4)


def merge(x):return x.permute(0,1,3,2,4).flatten(-2)


def rotate(x,frequencies,positions):
    if not frequencies:return x
    f=x.new_tensor(frequencies);size=len(f)*2
    if size>x.shape[-1]:raise ValueError('RoPE exceeds declared head width')
    angle=positions[:,None]*f[None,:];c=angle.cos();s=angle.sin()
    pair=x[...,-size:].reshape(*x.shape[:-1],len(f),2)
    a,b=pair.unbind(-1)
    rotated=torch.stack((a*c-b*s,a*s+b*c),-1).flatten(-2)
    return torch.cat((x[...,:-size],rotated),-1)


class Memory(nn.Module):
    def __init__(self,n,spec,stride):
        super().__init__();self.n=n;self.s=spec;self.stride=stride
        d=spec['width'];ds=spec['summary_width'];f=spec['ffn_width']
        self.family=spec.get('memory_family','kv');self.current=NodeLinear(n,d,ds)
        self.ffgate=NodeLinear(n,d,f);self.ffup=NodeLinear(n,d,f);self.ffdown=NodeLinear(n,f,d)
        if self.family=='kv':
            rank=spec['adapter_rank'];self.base=nn.Parameter(torch.randn(2*d,d)/math.sqrt(d))
            self.adapter_in=NodeLinear(n,d,rank);self.adapter_out=NodeLinear(n,rank,2*d)
            # Balanced low-rank delta; still independently learnable both factors.
            with torch.no_grad():self.adapter_out.weight.mul_(.1)
            self.query=NodeLinear(n,d,d);self.output=NodeLinear(n,d,d)
            if spec['memory_read']=='qry':
                self.read_q=NodeLinear(n,d,ds);self.read_k=NodeLinear(n,d,ds);self.read_v=NodeLinear(n,d,ds)
        elif self.family in ('affine','delta'):
            dk=spec['key_width'];dm=spec['value_width']
            self.key=NodeLinear(n,d,dk);self.value=NodeLinear(n,d,dm)
            self.retain=NodeLinear(n,d,1,True)
            if self.family=='delta':self.write=NodeLinear(n,d,1,True)
            self.full_q=nn.ModuleList([NodeLinear(n,d,dk) for _ in range(2)])
            self.output=nn.ModuleList([NodeLinear(n,dm,d) for _ in range(2)])
            if spec['memory_read']=='state':
                self.read_q=nn.ModuleList([NodeLinear(n,d,dk) for _ in range(2)])
                self.read_out=nn.ModuleList([NodeLinear(n,dm,ds) for _ in range(2)])
        else:
            h=spec['heads'];dk=spec['key_width'];dm=spec['value_width'];inner=h*dm
            self.h=h;self.dk=dk;self.dm=dm
            if self.family=='gated_delta':
                self.projection=NodeLinear(n,d,h*(2*dk+dm))
                self.retain=NodeLinear(n,d,h,True);self.write=NodeLinear(n,d,h,True)
                self.gate=NodeLinear(n,d,inner)
                conv_width=h*(2*dk+dm)
            elif self.family=='ssd':
                groups=spec['groups'];conv_width=inner+2*groups*dk
                self.projection=NodeLinear(n,d,2*inner+2*groups*dk+h)
                self.dt_bias=nn.Parameter(torch.zeros(n,h))
                self.skip=nn.Parameter(torch.ones(n,h))
            else:raise ValueError('unsupported memory family')
            self.rate_log=nn.Parameter(torch.zeros(n,h))
            self.conv=nn.Parameter(torch.randn(n,spec['conv_size'],conv_width)/math.sqrt(spec['conv_size']))
            self.conv_bias=nn.Parameter(torch.zeros(n,conv_width))
            self.output_scale=nn.Parameter(torch.ones(n,inner));self.output=NodeLinear(n,inner,d)
            if spec['memory_read']=='state':self.read_out=NodeLinear(n,inner,ds)

    def kv(self,u,w,x,p,read):
        s=self.s;n,b,l,d=x.shape;h=s['heads'];window=s['window'];beta=s['beta_short']
        kv=F.linear(x,self.base)+self.adapter_out(self.adapter_in(x));k,v=kv.chunk(2,-1)
        values,masses=summary(w,p,self.stride,beta,read,window)
        positions=(torch.arange(l,device=x.device,dtype=x.dtype)*self.stride+s['position_origin']-s['position_origin'])/s['position_scale']
        if s['memory_read']=='qry':
            q=rotate(self.read_q(x),s['read_frequencies'],positions)
            rk=rotate(self.read_k(k),s['read_frequencies'],positions);rv=self.read_v(v)
            score=(q@rk.transpose(-1,-2))/math.sqrt(s['summary_width'])
            values=[]
            for i,rate in enumerate((beta,0.)):
                a,mask,_=time_weights(w,self.stride,rate,window,read)
                values.append((weighted_softmax(score,a,mask)@rv)*masses[i].clamp_max(1).unsqueeze(-1))
        q=heads(self.query(x),h);kh=heads(k,h);vh=heads(v,h)
        q=torch.stack([rotate(q[:,:,j],s['full_frequencies'][j],positions) for j in range(h)],2)
        kh=torch.stack([rotate(kh[:,:,j],s['full_frequencies'][j],positions) for j in range(h)],2)
        score=(q@kh.transpose(-1,-2))/math.sqrt(d//h)
        a,mask,dt=time_weights(w,self.stride,0.,window,'+')
        slopes=x.new_tensor(s['distance_slopes']).reshape(1,1,h,1,1)
        score=score-slopes*(dt.clamp_min(0).to(x.dtype)/s['position_scale'])
        full=merge((weighted_softmax(score,a.unsqueeze(2),mask)@vh)*a.sum(-1).clamp_max(1)[:,:,None,:,None])
        content=u+s['alpha_att']*self.output(full)
        return values,masses,content

    def recurrent(self,u,w,x,p,read):
        s=self.s;n,b,l,d=x.shape;ds=s['summary_width'];rate=s['lambda_short']
        values,masses=summary(w,p,self.stride,rate,read)
        if self.family in ('affine','delta'):
            k=self.key(x);k=k/torch.linalg.vector_norm(k,dim=-1,keepdim=True).clamp_min(1)
            v=self.value(x);log_input=-w*F.softplus(self.retain(x).squeeze(-1))
            write=w if self.family=='affine' else w*torch.sigmoid(self.write(x).squeeze(-1))
            mix=torch.zeros_like(u);state_values=[]
            for a,passive in enumerate((rate,0.)):
                logg=log_input-passive*self.stride
                fq=self.full_q[a](x)
                r=dual_recurrence(k,v,fq,logg,write,delta=self.family=='delta')
                mix=mix+self.output[a](r)
                if s['memory_read']=='state':
                    if read=='0':mr=x.new_zeros(n,b,l,ds)
                    else:
                        rr=dual_recurrence(k,v,self.read_q[a](x),logg,write,delta=self.family=='delta',
                                           old=read=='-',log_input_gate=log_input)
                        mr=self.read_out[a](rr)
                    state_values.append(mr)
            if s['memory_read']=='state':values=state_values
        else:
            h=self.h;dk=self.dk;dm=self.dm;inner=h*dm
            raw=self.projection(x)
            if self.family=='gated_delta':
                conv=F.silu(causal_conv(raw,w,self.conv,self.conv_bias,self.stride,s['conv_stride']))
                per=conv.reshape(n,b,l,h,2*dk+dm).permute(0,1,3,2,4)
                q,k,v=per.split((dk,dk,dm),-1)
                k=k*torch.rsqrt(k.square().sum(-1,keepdim=True)+s['qk_epsilon'])
                q=q*torch.rsqrt(q.square().sum(-1,keepdim=True)+s['qk_epsilon'])/math.sqrt(dk)
                dt=F.softplus(self.retain(x)).permute(0,1,3,2)
                write=w.unsqueeze(2)*torch.sigmoid(self.write(x)).permute(0,1,3,2)
                gate=self.gate(x)
            else:
                g=s['groups'];gate,convraw,dt_raw=raw.split((inner,inner+2*g*dk,h),-1)
                conv=F.silu(causal_conv(convraw,w,self.conv,self.conv_bias,self.stride,s['conv_stride']))
                vraw,kr,qr=conv.split((inner,g*dk,g*dk),-1)
                v=heads(vraw,h);k=heads(kr,g).repeat_interleave(h//g,dim=2);q=heads(qr,g).repeat_interleave(h//g,dim=2)
                dt=F.softplus(dt_raw+self.dt_bias[:,None,None,:]).permute(0,1,3,2)
                write=w.unsqueeze(2)*dt
            log_input=-w.unsqueeze(2)*self.rate_log.exp()[:,None,:,None]*dt
            rates=x.new_tensor(s['matrix_rates']).reshape(1,1,h,1)
            logg=log_input-rates*self.stride
            r=dual_recurrence(k,v,q,logg,write,delta=self.family=='gated_delta')
            if s['memory_read']=='state':
                rr=r if read=='+' else (torch.zeros_like(r) if read=='0' else
                    dual_recurrence(k,v,q,logg,write,delta=self.family=='gated_delta',old=True,log_input_gate=log_input))
                values=[values[0],self.read_out(merge(rr))]
            if self.family=='gated_delta':
                normalized=norm(r,s['output_epsilon'])
                mix=self.output(merge(normalized)*self.output_scale[:,None,None,:]*F.silu(gate))
            else:
                y=merge(r+self.skip[:,None,:,None,None]*v)*F.silu(gate)
                mix=self.output(norm(y,s['output_epsilon'])*self.output_scale[:,None,None,:])
        return values,masses,u+s['alpha_mem']*mix

    def forward(self,u,r,read='+'):
        w=r.clamp(0,1);x=norm(u,self.s['epsilon']);p=self.current(x)
        values,masses,content=self.kv(u,w,x,p,read) if self.family=='kv' else self.recurrent(u,w,x,p,read)
        f=norm(content,self.s['epsilon'])
        content=content+self.s['alpha_ff']*self.ffdown(F.silu(self.ffgate(f))*self.ffup(f))
        return dict(x=x,p=p,memory=values,mass=masses,content=content)
