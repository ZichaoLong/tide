#include "extra_full_vjp.h"
#include <algorithm>
#include <set>
#include <stdexcept>

namespace tide::device_online {
void append_lh_full_vjp(CannProgram& p,const FullTape& t,const at::Tensor& gradient,const at::Tensor& connected,
    FullVjp& out,const at::Tensor& error,int64_t max_rows,int64_t budget) {
  const auto& e=t.extra;if(!e.lh_kinds.defined())return;
  const auto device=t.values.device();const int64_t capacity=t.values.size(0),nodes=t.kinds.size(0),d=t.width;
  const long double fixed=8.L*nodes*d+4.L*capacity*d+8.L*(nodes+1)+256.L,row=96.L*d+160.L;
  if(budget<1||fixed+row>budget)throw std::invalid_argument("one LH Full VJP row exceeds tensor budget");
  const int64_t chunk=std::min<int64_t>({capacity,max_rows,static_cast<int64_t>((budget-fixed)/row)});
  full_extra_tensor(e.lh_kinds,device,at::kLong,{nodes});full_extra_tensor(e.lh_weights,device,at::kFloat,{nodes+1,d});
  full_extra_tensor(e.lh_biases,device,at::kFloat,{nodes+1,d});
  std::set<int64_t> groups(e.lh_groups.begin(),e.lh_groups.end());
  if(groups.empty()||groups.size()!=e.lh_groups.size()||*groups.begin()<1||*groups.rbegin()>9)
    throw std::invalid_argument("invalid LH Full static VJP groups");
  auto floats=t.values.options();out.extra.lh_weights=at::empty({nodes,d},floats);out.extra.lh_biases=at::empty_like(out.extra.lh_weights);
  p.zero(out.extra.lh_weights);p.zero(out.extra.lh_biases);ExtraFullRows r(p,t,out,chunk);
  auto mapping=at::arange(nodes+1,at::kLong).to(device);
  auto activated=at::empty_like(r.x),normalized=at::empty_like(r.x),weight=at::empty_like(r.x),dnorm=at::empty_like(r.x);
  auto product=at::empty_like(r.x),correction=at::empty_like(r.x),dact=at::empty_like(r.x),dx=at::empty_like(r.x);
  auto dw=at::empty_like(r.x),db=at::empty_like(r.x);
  auto inv=at::empty({chunk,1},floats),sum=at::empty_like(inv),mean=at::empty_like(inv),dsum=at::empty_like(inv),dmean=at::empty_like(inv);
  auto negative_mean=at::full({1},-1.f/static_cast<float>(d),floats);
  for(auto kind:groups) {
    const auto act=(kind-1)/3,norm=(kind-1)%3;
    p.zero(r.cursor);auto head=p.label(),body=p.label(),done=p.label();p.mark(head);
    r.plan(p,t,connected,out,e.lh_kinds,mapping,kind,nodes,false,error);p.branch(r.branch,{done,body});p.mark(body);
    r.payload(p,t,gradient,out,false,error);
    if(act==0)p.relu(r.x,activated);else if(act==1)p.silu(r.x,activated);else p.copy(activated,r.x);
    if(norm) {
      if(norm==1)p.rms_norm(activated,1e-7,normalized,inv);else p.layer_norm(activated,1e-5,normalized,inv);
      p.index_select(e.lh_weights,0,r.parameters,weight);p.multiply(r.gradient,weight,dnorm);
      p.multiply(r.gradient,normalized,dw);
      if(norm==2)p.copy(db,r.gradient);else p.zero(db);
      // z is unit-affine normalized activation. Both formulas preserve the
      // declared epsilon and use the actual CANN FP32 reciprocal stddev.
      p.multiply(dnorm,normalized,product);p.sum(product,1,true,sum);p.multiply(sum,negative_mean,mean);
      p.multiply(normalized,mean,correction);p.add(correction,dnorm);
      if(norm==2){p.sum(dnorm,1,true,dsum);p.multiply(dsum,negative_mean,dmean);p.add(correction,dmean);}
      p.multiply(correction,inv,dact);
      r.reduce(p,dw,db,{},out.extra.lh_weights,out.extra.lh_biases,{},error);
    }else p.copy(dact,r.gradient);
    if(act==0)p.relu_backward(dact,r.x,dx);else if(act==1)p.silu_backward(dact,r.x,dx);else p.copy(dx,dact);
    p.index_copy(r.output,0,r.destination,dx);p.branch(r.branch,{head});p.mark(done);
  }
  p.copy(out.comparison,r.output.narrow(0,0,capacity));
}
} // namespace tide::device_online
