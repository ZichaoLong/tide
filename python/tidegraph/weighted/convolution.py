"""0770593 causal-convolution §1–3: fixed logical slots, never event compaction.

Cache is the exact saved finite map, not its effective-state projection. GDN
and SSD consume these primitives; this module does not schedule graph events.
"""
from dataclasses import dataclass
import torch
from ..coordinates import integers
from .config import positive_int
from .tensors import parameter


@dataclass(frozen=True)
class ConvCache:
    times: torch.Tensor
    values: torch.Tensor

    @classmethod
    def empty(cls, reference):
        return cls(torch.empty(0,dtype=torch.int64,device=reference.device),
                   reference.new_empty((0,reference.numel())))

    def validate(self, width, horizon):
        if (self.times.dtype != torch.int64 or self.times.ndim != 1 or self.times.requires_grad
                or self.values.shape != (len(self.times),width) or self.values.device != self.times.device
                or not self.values.is_floating_point() or len(self.times)>horizon+1):
            raise ValueError("invalid fixed-time convolution cache layout")
        if ((self.times<0).any() or (self.times[1:]<=self.times[:-1]).any()
                or not torch.isfinite(self.values).all()):
            raise ValueError("invalid fixed-time convolution cache values/times")


class FixedTimeConv(torch.nn.Module):
    def __init__(self,width,size,stride,*,generator,dtype):
        super().__init__()
        for name,value in (("width",width),("conv_size",size),("conv_stride",stride)):
            positive_int(name,value)
        self.width,self.size,self.stride=width,size,stride
        self.horizon=(size-1)*stride
        positive_int("conv horizon",self.horizon,0)
        if self.horizon == 2**63-1:
            raise ValueError("convolution horizon capacity overflows int64")
        self.weight=parameter((size,width),generator,dtype)
        self.bias=parameter((width,),generator,dtype)

    def _validate(self,cache,time,z):
        integers("convolution time",time)
        if time<0 or z.shape!=(self.width,) or (z.dtype,z.device)!=(self.weight.dtype,self.weight.device):
            raise ValueError("invalid convolution query dtype/device/shape/time")
        cache.validate(self.width,self.horizon)
        if (cache.values.dtype,cache.values.device)!=(z.dtype,z.device) or not torch.isfinite(z).all():
            raise ValueError("invalid convolution cache/query value")

    def forward(self,cache,time,z):
        self._validate(cache,time,z)
        result=self.bias+self.weight[0]*z
        if self.size==1 or not len(cache.times):
            return result
        taps=time-self.stride*torch.arange(1,self.size,device=z.device,dtype=torch.int64)
        # Gather by absolute identity; invalid/negative taps contribute zero.
        # Equal-mask matmul uses the ordinary VJP of the packed cache tensor.
        matches=(taps.unsqueeze(1)==cache.times.unsqueeze(0)).to(z.dtype)
        history=matches@cache.values
        return result+(self.weight[1:]*history).sum(0)

    def write(self,cache,time,weight,z):
        self._validate(cache,time,z)
        if (not isinstance(weight,torch.Tensor) or weight.shape!=()
                or (weight.dtype,weight.device)!=(z.dtype,z.device) or not torch.isfinite(weight)
                or weight<0 or weight>1):
            raise ValueError("convolution write weight must lie in [0,1]")
        if weight==0:
            return cache
        if len(cache.times) and time<=int(cache.times[-1]):
            raise ValueError("convolution writes require increasing logical time")
        mask=(cache.times>=max(0,time-self.horizon))&(cache.times<time)
        return ConvCache(torch.cat((cache.times[mask],torch.tensor([time],device=z.device,dtype=torch.int64))),
                         torch.cat((cache.values[mask],(weight*z).unsqueeze(0))))

    def effective(self,cache,time):
        """Explicit property-comparison projection; never a checkpoint decoder."""
        integers("convolution projection time",time)
        if time<0:
            raise ValueError("negative convolution projection time")
        cache.validate(self.width,self.horizon)
        start=max(0,time-self.horizon)
        # arange's exclusive endpoint must not become INT64_MAX + 1.
        slots=torch.arange(time-start+1,device=cache.times.device,dtype=torch.int64)+start
        matches=(slots.unsqueeze(1)==cache.times.unsqueeze(0)).to(cache.values.dtype)
        return slots,matches@cache.values
