#include "control_vjp.h"
#include "cann_api.h"
#include "aclrtlaunch_tide_control_plan.h"
#include "aclrtlaunch_tide_control_payload.h"
#include "aclrtlaunch_tide_control_read_reduce.h"
#include "aclrtlaunch_tide_control_merge.h"
#include <ATen/core/grad_mode.h>
#include <cmath>
#include <stdexcept>

namespace tide::device_online {
namespace {
uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}
void tensor(const at::Tensor& x,at::Device d,at::ScalarType type,at::IntArrayRef shape) {
  if(!x.defined()||x.device()!=d||x.scalar_type()!=type||x.sizes()!=shape||!x.is_contiguous()||x.requires_grad())
    throw std::invalid_argument("invalid control VJP buffer");
}
}
ControlVjp append_control_vjp(CannProgram& p,const Graph& g,const StateTape& t,const ControlTape& c,
    const at::Tensor& count,const at::Tensor& range,const at::Tensor& gradient,const at::Tensor& on,
    const at::Tensor& error,int64_t budget) {
  if(at::GradMode::is_enabled()||!gradient.defined()||gradient.dim()!=2||!t.metadata.defined()||t.metadata.dim()!=2)
    throw std::invalid_argument("control VJP requires no-grad actual device tape");
  const auto d=gradient.device();const int64_t capacity=t.metadata.size(0),width=gradient.size(1),nodes=g.nodes.size();
  if(d.type()!=c10::DeviceType::PrivateUse1||capacity<1||width<1||nodes<1||t.samples<1||budget<1
      ||(c.mode!=1&&c.mode!=2)||!std::isfinite(c.zeta)
      ||32.L*capacity*width+96.L*capacity+4.L*nodes*width+64.L*nodes+512>budget)
    throw std::invalid_argument("control VJP tensor budget or mode unavailable");
  int64_t buckets=1;while(buckets<2*capacity)buckets*=2;
  tensor(t.metadata,d,at::kLong,{capacity,13});tensor(t.values,d,at::kFloat,{capacity,5*width+2});
  tensor(c.raw_full,d,at::kFloat,{capacity,width});tensor(c.read,d,at::kFloat,{nodes,width});
  tensor(count,d,at::kLong,{1});tensor(range,d,at::kLong,{2});tensor(gradient,d,at::kFloat,{capacity,width});
  tensor(on,d,at::kBool,{capacity});tensor(error,d,at::kInt,{1});
  std::vector<int64_t> settings;
  for(const auto& n:g.nodes) {
    const auto& r=g.regions.at(n.region);
    if((r.read_mode!="content"&&r.read_mode!="old"&&r.read_mode!="proposal")
        ||(n.readout!="linear-v1"&&n.readout!="norm-fp32-v1"))throw std::invalid_argument("control Read adjoint unavailable");
    settings.insert(settings.end(),{n.region,n.identity?-1:r.read_mode=="content"?0:r.read_mode=="old"?1:2,
      n.readout=="norm-fp32-v1"});
  }
  auto config=at::tensor(settings,at::kLong).reshape({nodes,3}).to(d);
  auto f=gradient.options(),l=t.metadata.options(),b=on.options();
  ControlVjp out{at::empty_like(gradient),at::empty({capacity,5,width},f),at::empty({capacity,5},b),
    at::empty({nodes,width},f),at::empty({nodes},b)};
  auto hash=at::empty({buckets},l),frame=at::empty({capacity},l),next=at::empty_like(frame);
  auto owner_head=at::empty({nodes},l),owner_next=at::empty_like(frame),read_on=at::empty({capacity},b);
  auto products=at::empty_like(gradient),dp=at::empty({capacity},f),ds=at::empty_like(dp),partials=at::empty_like(gradient);
  for(const auto& x:{out.fresh,out.events,out.connected,out.read,out.read_connected,products,dp,ds,partials,read_on})p.zero(x);
  auto plan=[&](int64_t mode) {
    p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_control_plan)(1,stream,
      ptr(t.metadata),ptr(t.values),ptr(count),ptr(range),ptr(config),ptr(on),ptr(hash),ptr(frame),ptr(next),ptr(owner_head),ptr(owner_next),
      ptr(dp),ptr(ds),ptr(read_on),ptr(out.connected),ptr(out.read_connected),ptr(error),capacity,width,nodes,t.samples,buckets,mode,c.mode,float(c.zeta)),
      "link complete control frames and declared Read connections");},
      {t.metadata,t.values,count,range,config,on,hash,frame,next,owner_head,owner_next,dp,ds,read_on,out.connected,out.read_connected,error});
  };
  auto payload=[&](int64_t mode) {
    p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_control_payload)(32,stream,
      ptr(t.metadata),ptr(t.values),ptr(c.raw_full),ptr(c.read),ptr(count),ptr(range),ptr(config),ptr(gradient),ptr(on),ptr(read_on),ptr(ds),
      ptr(out.fresh),ptr(out.events),ptr(products),ptr(partials),ptr(error),capacity,width,mode,c.mode),"packed Emit and Read adjoints");},
      {t.metadata,t.values,c.raw_full,c.read,count,range,config,gradient,on,read_on,ds,out.fresh,out.events,products,partials,error});
  };
  plan(0);payload(0);p.sum(products,1,false,dp);plan(1);payload(1);
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_control_read_reduce)(32,stream,
    ptr(owner_head),ptr(owner_next),ptr(read_on),ptr(partials),ptr(out.read),ptr(error),nodes,width),"ordered Read parameter partials");},
    {owner_head,owner_next,read_on,partials,out.read,error});
  return out;
}
void append_control_merge(CannProgram& p,const ControlVjp& x,const StateCotangents& cot,const at::Tensor& error) {
  p.add(cot.events,x.events);
  append_connection_union(p,x.connected,cot.connected,error);
}
void append_connection_union(CannProgram& p,const at::Tensor& source,const at::Tensor& target,const at::Tensor& error) {
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_control_merge)(1,stream,
    ptr(source),ptr(target),ptr(error),source.numel()),"merge structural control cotangents");},
    {source,target,error});
}
} // namespace tide::device_online
