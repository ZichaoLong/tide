#include "extra_full_vjp.h"
#include <algorithm>
#include <stdexcept>

namespace tide::device_online {
void append_swiglu_vjp(CannProgram& p,const FullTape& t,const at::Tensor& gradient,const at::Tensor& connected,
    FullVjp& out,const at::Tensor& error,int64_t max_rows,int64_t budget) {
  const auto& e=t.extra;if(!e.swiglu_kinds.defined())return;
  const auto device=t.values.device();const int64_t capacity=t.values.size(0),nodes=t.kinds.size(0),d=t.width;
  if(!e.gate.defined()||e.gate.dim()!=3||e.gate.size(0)<2)throw std::invalid_argument("invalid SwiGLU VJP owner bank");
  const auto payload=e.gate.scalar_type();const bool half=payload==at::kHalf;
  if(payload!=at::kFloat&&!half)throw std::invalid_argument("SwiGLU VJP requires FP32/FP16 forward parameters");
  const auto owners=e.gate.size(0)-1;
  const long double fixed=24.L*owners*d*d+4.L*capacity*d+128.L;
  // Three matrices, three transposes, three partials; reusable vector work.
  const long double row=(72.L+12.L*half)*d*d+(128.L+18.L*half)*d+128.L;
  if(budget<1||fixed+row>budget)throw std::invalid_argument("one SwiGLU VJP row exceeds tensor budget");
  const int64_t chunk=std::min<int64_t>({capacity,max_rows,static_cast<int64_t>((budget-fixed)/row)});
  full_extra_tensor(e.swiglu_kinds,device,at::kLong,{nodes});full_extra_tensor(e.swiglu_mapping,device,at::kLong,{nodes+1});
  full_extra_tensor(e.gate,device,payload,{owners+1,d,2*d});full_extra_tensor(e.up,device,payload,{owners+1,d,2*d});
  full_extra_tensor(e.down,device,payload,{owners+1,2*d,d});
  auto floats=t.values.options();out.extra.gate=at::empty({owners,d,2*d},floats);out.extra.up=at::empty_like(out.extra.gate);
  out.extra.down=at::empty({owners,2*d,d},floats);
  for(const auto& x:{out.extra.gate,out.extra.up,out.extra.down})p.zero(x);
  ExtraFullRows r(p,t,out,chunk);
  auto gate=at::empty({chunk,d,2*d},floats),up=at::empty_like(gate),ddown=at::empty_like(gate);
  auto down=at::empty({chunk,2*d,d},floats),gt=at::empty_like(down),ut=at::empty_like(down);
  auto dg=at::empty_like(gate),du=at::empty_like(gate),dd=at::empty_like(down);
  auto a=at::empty({chunk,1,2*d},floats),b=at::empty_like(a),activated=at::empty_like(a),product=at::empty_like(a);
  auto dp=at::empty_like(a),da=at::empty_like(a),db=at::empty_like(a),temp=at::empty_like(a);
  auto dx=at::empty({chunk,1,d},floats),dx2=at::empty_like(dx);
  const auto x=r.x.view({chunk,1,d}),g=r.gradient.view({chunk,1,d});
  const auto forward=floats.dtype(payload);
  auto fx=half?at::empty({chunk,1,d},forward):x;
  auto fg=half?at::empty({chunk,d,2*d},forward):gate,fu=half?at::empty({chunk,d,2*d},forward):up;
  auto fd=half?at::empty({chunk,2*d,d},forward):down;
  auto fa=half?at::empty({chunk,1,2*d},forward):a,fb=half?at::empty({chunk,1,2*d},forward):b;
  auto fact=half?at::empty({chunk,1,2*d},forward):activated,fp=half?at::empty({chunk,1,2*d},forward):product;
  p.zero(r.cursor);auto head=p.label(),body=p.label(),done=p.label();p.mark(head);
  r.plan(p,t,connected,out,e.swiglu_kinds,e.swiglu_mapping,1,owners,true,error);p.branch(r.branch,{done,body});p.mark(body);
  r.payload(p,t,gradient,out,true,error);
  p.index_select(e.gate,0,r.parameters,fg);p.index_select(e.up,0,r.parameters,fu);p.index_select(e.down,0,r.parameters,fd);
  if(half)p.cast(x,fx);
  p.batch_matmul(fx,fg,fa);p.batch_matmul(fx,fu,fb);p.silu(fa,fact);p.multiply(fact,fb,fp);
  if(half) {
    p.cast(fg,gate);p.cast(fu,up);p.cast(fd,down);
    p.cast(fa,a);p.cast(fb,b);p.cast(fact,activated);p.cast(fp,product);
  }
  p.permute(down,{0,2,1},ddown);p.batch_matmul(g,ddown,dp);
  p.multiply(dp,b,temp);p.silu_backward(temp,a,da);p.multiply(dp,activated,db);
  p.permute(gate,{0,2,1},gt);p.permute(up,{0,2,1},ut);
  p.batch_matmul(da,gt,dx);p.batch_matmul(db,ut,dx2);p.add(dx,dx2);
  p.batch_matmul(r.x.view({chunk,d,1}),da,dg);p.batch_matmul(r.x.view({chunk,d,1}),db,du);
  p.batch_matmul(product.view({chunk,2*d,1}),g,dd);
  p.index_copy(r.output,0,r.destination,dx.view({chunk,d}));r.reduce(p,dg,du,dd,out.extra.gate,out.extra.up,out.extra.down,error);
  p.branch(r.branch,{head});p.mark(done);p.copy(out.comparison,r.output.narrow(0,0,capacity));
}
} // namespace tide::device_online
