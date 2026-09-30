#include "extra_full_vjp.h"
#include "cann_api.h"
#include "aclrtlaunch_tide_extra_full_plan.h"
#include "aclrtlaunch_tide_extra_full_payload.h"
#include "aclrtlaunch_tide_extra_full_reduce.h"
#include <stdexcept>

namespace tide::device_online {
namespace {uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}}
void full_extra_tensor(const at::Tensor& x,at::Device device,at::ScalarType type,at::IntArrayRef shape) {
  if(!x.defined()||x.device()!=device||x.scalar_type()!=type||x.sizes()!=shape||!x.is_contiguous()||x.requires_grad())
    throw std::invalid_argument("invalid extended Full VJP bank");
}
ExtraFullRows::ExtraFullRows(CannProgram& p,const FullTape& t,const FullVjp& out,int64_t rows):chunk(rows) {
  auto longs=t.kinds.options(),floats=t.values.options();
  source=at::empty({chunk},longs);parameters=at::empty_like(source);destination=at::empty_like(source);owners=at::empty_like(source);
  owner_count=at::empty({1},longs);cursor=at::empty_like(owner_count);branch=at::empty({1},longs.dtype(at::kInt));
  x=at::empty({chunk,t.width},floats);gradient=at::empty_like(x);
  output=at::empty({t.values.size(0)+chunk,t.width},floats);p.zero(output);p.copy(output.narrow(0,0,t.values.size(0)),out.comparison);
}
void ExtraFullRows::plan(CannProgram& p,const FullTape& t,const at::Tensor& connected,FullVjp& out,
    const at::Tensor& kinds,const at::Tensor& mapping,int64_t kind,int64_t parameter_count,bool residual,const at::Tensor& error) const {
  auto w=*this;const auto capacity=t.values.size(0),nodes=t.kinds.size(0);
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_extra_full_plan)(1,stream,
    ptr(t.metadata),ptr(t.count),ptr(kinds),ptr(mapping),ptr(connected),ptr(out.content_connected),ptr(out.comparison_connected),
    ptr(out.parameter_connected),ptr(w.cursor),ptr(w.source),ptr(w.parameters),ptr(w.destination),ptr(w.owners),ptr(w.owner_count),
    ptr(w.branch),ptr(out.chunks),ptr(error),capacity,nodes,w.chunk,kind,parameter_count,int64_t(residual)),"pack connected extended Full rows");},
    {t.metadata,t.count,kinds,mapping,connected,out.content_connected,out.comparison_connected,out.parameter_connected,cursor,
     source,parameters,destination,owners,owner_count,branch,out.chunks,error});
}
void ExtraFullRows::payload(CannProgram& p,const FullTape& t,const at::Tensor& upstream,FullVjp& out,bool residual,const at::Tensor& error) const {
  auto w=*this;
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_extra_full_payload)(32,stream,
    ptr(t.values),ptr(upstream),ptr(w.source),ptr(w.x),ptr(w.gradient),ptr(out.content),ptr(error),
    t.values.size(0),t.width,w.chunk,int64_t(residual)),"gather extended Full VJP payload");},
    {t.values,upstream,source,x,gradient,out.content,error});
}
void ExtraFullRows::reduce(CannProgram& p,const at::Tensor& a,const at::Tensor& b,const at::Tensor& c,
    const at::Tensor& oa,const at::Tensor& ob,const at::Tensor& oc,const at::Tensor& error) const {
  auto w=*this;const int64_t na=a.numel()/chunk,nb=b.numel()/chunk,nc=c.defined()?c.numel()/chunk:0;
  const auto input_c=c.defined()?c:a,output_c=oc.defined()?oc:oa;
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_extra_full_reduce)(32,stream,
    ptr(w.owners),ptr(w.owner_count),ptr(w.parameters),ptr(a),ptr(b),ptr(input_c),ptr(oa),ptr(ob),ptr(output_c),ptr(error),
    na,nb,nc,w.chunk),"ordered extended Full parameter partials");},
    {owners,owner_count,parameters,a,b,input_c,oa,ob,output_c,error});
}
} // namespace tide::device_online
