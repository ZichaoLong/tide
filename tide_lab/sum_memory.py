"""Specialized inclusive KV Mem.sum, with no per-node L-by-L weights.

Matrix uses the existing explicit positive-floor decay kernel. Blocked uses
two shared small matrices and composes decay factors across blocks; floating
underflow/rounding can differ and is checked separately from learning.
"""
import torch
from torch.nn import functional as F


def decay_sum(value, decay, *, rate, stride, implementation='matrix', block=32):
    if implementation=='matrix':
        return F.linear(value.transpose(-1,-2),decay).transpose(-1,-2)
    if implementation!='blocked':raise ValueError('unknown Mem.sum implementation')
    length,width=value.shape[-2:];chunks=(length+block-1)//block
    x=F.pad(value,(0,0,0,chunks*block-length)).reshape(*value.shape[:-2],chunks,block,width)
    pos=torch.arange(block,device=value.device,dtype=value.dtype)
    age=pos[:,None]-pos[None,:]
    local=(-rate*stride*age.clamp_min(0)).exp()*(age>=0)
    within=torch.matmul(local,x)
    ci=torch.arange(chunks,device=value.device,dtype=value.dtype)
    gap=ci[:,None]-ci[None,:]-1
    cross=(-rate*stride*block*gap.clamp_min(0)).exp()*(gap>=0)
    carried=torch.matmul(cross,within[...,-1,:])
    offset=(-rate*stride*(pos+1)).exp()
    result=within+carried.unsqueeze(-2)*offset[:,None]
    return result.reshape(*value.shape[:-2],chunks*block,width)[...,:length,:]


def sum_read(w,p,masses,decay,*,rate,stride,implementation='matrix'):
    weighted=w.unsqueeze(-1)*p
    short=decay_sum(weighted,decay,rate=rate,stride=stride,implementation=implementation)
    long=weighted.cumsum(-2)
    return [num/mass.clamp_min(1).unsqueeze(-1) for num,mass in zip((short,long),masses)]
