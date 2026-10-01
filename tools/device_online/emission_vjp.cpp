#include "emission_vjp.h"
#include "cann_api.h"
#include "aclrtlaunch_tide_emission_reverse_links.h"
#include "aclrtlaunch_tide_emission_vjp_plan.h"
#include "aclrtlaunch_tide_emission_vjp_payload.h"
#include "aclrtlaunch_tide_full_vjp_reduce.h"
#include <ATen/core/grad_mode.h>
#include <algorithm>
#include <stdexcept>
namespace tide::device_online {
namespace {
uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}
void tensor(const at::Tensor& x,at::Device d,at::ScalarType type,at::IntArrayRef shape) {
  if(!x.defined()||x.device()!=d||x.scalar_type()!=type||x.sizes()!=shape||!x.is_contiguous()||x.requires_grad())
    throw std::invalid_argument("invalid emission reverse tape/buffer");
}
}
EmissionReverse prepare_emission_reverse(CannProgram& p,const ReverseTape& t,const ReverseLinks& links,
                                         const at::Tensor& error,int64_t budget) {
  if(at::GradMode::is_enabled()||!t.graph||t.control.mode!=0||!t.emission.weights.defined())
    throw std::invalid_argument("slot-affine reverse requires actual HARD emission tape");
  const auto d=t.state.metadata.device();const auto dtype=t.emission.weights.scalar_type();
  const auto width=t.full.width,capacity=t.state.metadata.size(0),total=links.messages.size(0);
  const auto nodes=int64_t(t.graph->nodes.size()),emissions=t.emission.metadata.size(0);
  int64_t parameters=0;std::vector<int64_t> mapping;
  for(int64_t n=0;n<nodes;++n)for(auto j=t.graph->outgoing_ports.offsets[n];j<t.graph->outgoing_ports.offsets[n+1];++j)
    mapping.push_back(!t.graph->nodes[n].identity&&t.graph->nodes[n].emission=="slot_affine"?parameters++:-1);
  int64_t buckets=1;while(buckets<2*emissions)buckets*=2;
  const long double bytes=4.L*parameters*(width*static_cast<long double>(width)+width)+parameters
    +16.L*total+8.L*(buckets+nodes+mapping.size()+1)+256;
  if(d.type()!=c10::DeviceType::PrivateUse1||parameters<1||capacity<1||emissions<1||width<1||budget<1||bytes>budget
      ||(dtype!=at::kFloat&&dtype!=at::kHalf))throw std::invalid_argument("emission reverse tensor budget exceeded");
  tensor(t.emission.metadata,d,at::kLong,{emissions,6});tensor(t.emission.values,d,at::kFloat,{emissions,width});
  tensor(t.emission.count,d,at::kLong,{1});tensor(t.emission.weights,d,dtype,{parameters+1,width,width});
  tensor(t.emission.biases,d,dtype,{parameters+1,width});tensor(error,d,at::kInt,{1});
  auto offsets=at::tensor(t.graph->outgoing_ports.offsets,at::kLong).to(d),map=at::tensor(mapping,at::kLong).to(d);
  auto hash=at::empty({buckets},offsets.options());
  auto f=t.emission.values.options(),b=f.dtype(at::kBool),l=offsets.options();
  EmissionReverse out{{at::empty({parameters,width,width},f),at::empty({parameters,width},f),
    at::empty({parameters},b),at::empty({1},l)},at::empty({total},l),at::empty({total},l)};
  for(const auto& x:{out.gradient.weights,out.gradient.biases,out.gradient.connected,out.gradient.chunks})p.zero(x);
  const int64_t scale_offset=t.graph->inputs.size()+t.graph->edges.size();
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_emission_reverse_links)(1,stream,
    ptr(t.state.metadata),ptr(t.state.count),ptr(t.emission.metadata),ptr(t.emission.count),ptr(links.messages),ptr(links.valid),
    ptr(offsets),ptr(map),ptr(hash),ptr(out.rows),ptr(out.parameters),ptr(error),capacity,emissions,total,nodes,t.state.samples,
    scale_offset,buckets,parameters),"associate physical messages with actual unscaled emission slots");},
    {t.state.metadata,t.state.count,t.emission.metadata,t.emission.count,links.messages,links.valid,offsets,map,hash,out.rows,out.parameters,error});
  return out;
}
void append_emission_reverse(CannProgram& p,const ReverseTape& t,const ReverseLinks& links,const EmissionReverse& r,
    const at::Tensor& messages,const at::Tensor& on,const at::Tensor& range,const at::Tensor& full_gradient,
    const at::Tensor& error,int64_t max_rows,int64_t budget) {
  const int64_t capacity=t.state.metadata.size(0),total=links.messages.size(0),width=t.full.width,parameters=r.gradient.weights.size(0);
  const bool half=t.emission.weights.scalar_type()==at::kHalf;
  const long double fixed=4.L*(2*total+capacity+3.L)*width+256;
  const long double row=(12.L+2.L*half)*width*width+24.L*width+64;
  if(max_rows<1||budget<1||fixed+row>budget)throw std::invalid_argument("one emission adjoint row exceeds tensor budget");
  const int64_t chunk=std::min<int64_t>({total,max_rows,static_cast<int64_t>((budget-fixed)/row)});
  auto f=t.emission.values.options(),l=t.state.metadata.options();
  auto upstream=at::empty({total+1,width},f),projected=at::empty({total+chunk,width},f),full=at::empty({capacity+1,width},f);
  p.zero(upstream);p.zero(projected);p.zero(full);p.copy(full.narrow(0,0,capacity),t.full_values);
  auto source=at::empty({chunk},l),events=at::empty_like(source),param=at::empty_like(source),dest=at::empty_like(source),owners=at::empty_like(source);
  auto owner_count=at::empty({1},l),cursor=at::empty({2},l),reset=at::full({2},-1,l),branch=at::empty_like(error);
  auto x=at::empty({chunk,width},f),dy=at::empty_like(x),dx=at::empty_like(x);
  auto weights=at::empty({chunk,width,width},f),transposed=at::empty_like(weights),dw=at::empty_like(weights);
  auto payload_weights=half?at::empty({chunk,width,width},f.dtype(at::kHalf)):weights;
  auto payload=[&](int64_t mode) {
    p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_emission_vjp_payload)(32,stream,
      ptr(links.messages),ptr(links.producer_head),ptr(links.producer_next),ptr(r.parameters),ptr(links.scales),ptr(on),ptr(messages),
      ptr(range),ptr(upstream),ptr(projected),ptr(full_gradient),ptr(error),width,mode),"packed emission input adjoints");},
      {links.messages,links.producer_head,links.producer_next,r.parameters,links.scales,on,messages,range,upstream,projected,full_gradient,error});
  };
  payload(0);p.copy(cursor,reset);
  auto head=p.label(),body=p.label(),done=p.label();p.mark(head);
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_emission_vjp_plan)(1,stream,
    ptr(links.messages),ptr(links.producer_head),ptr(links.producer_next),ptr(r.parameters),ptr(on),ptr(range),ptr(cursor),ptr(source),ptr(events),
    ptr(param),ptr(dest),ptr(owners),ptr(owner_count),ptr(r.gradient.connected),ptr(branch),ptr(r.gradient.chunks),ptr(error),
    capacity,total,parameters,chunk),"pack connected physical projection adjoints");},
    {links.messages,links.producer_head,links.producer_next,r.parameters,on,range,cursor,source,events,param,dest,owners,owner_count,r.gradient.connected,branch,r.gradient.chunks,error});
  p.branch(branch,{done,body});p.mark(body);
  p.index_select(full,0,events,x);p.index_select(upstream,0,source,dy);p.index_select(t.emission.weights,0,param,payload_weights);
  if(half)p.cast(payload_weights,weights);
  p.permute(weights,{0,2,1},transposed);p.batch_matmul(dy.reshape({chunk,1,width}),transposed,dx.reshape({chunk,1,width}));
  p.batch_matmul(x.reshape({chunk,width,1}),dy.reshape({chunk,1,width}),dw);p.index_copy(projected,0,dest,dx);
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_full_vjp_reduce)(32,stream,
    ptr(owners),ptr(owner_count),ptr(param),ptr(dw),ptr(dy),ptr(r.gradient.weights),ptr(r.gradient.biases),ptr(error),width,chunk),
    "ordered per-slot projection parameter reduction");},{owners,owner_count,param,dw,dy,r.gradient.weights,r.gradient.biases,error});
  p.branch(branch,{head});p.mark(done);payload(1);
}
} // namespace tide::device_online
