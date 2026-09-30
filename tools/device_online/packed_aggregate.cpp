#include "packed_aggregate.h"
#include "cann_api.h"
#include "aclrtlaunch_tide_aggregate_plan.h"
#include "aclrtlaunch_tide_aggregate_apply.h"
#include <algorithm>
#include <stdexcept>

namespace tide::device_online {
int64_t aggregate_kind(const std::string& name) {
  if(name=="sum")return 0;if(name=="mean")return 1;if(name=="weighted_mean")return 2;
  if(name=="active_softmax")return 3;if(name=="all_softmax")return 4;
  throw std::invalid_argument("device Aggregate profile unavailable");
}
int64_t aggregate_slots(const Graph& graph) {
  int64_t slots=1;
  for(size_t i=0;i<graph.nodes.size();++i)if(aggregate_kind(graph.nodes[i].aggregation))
    slots=std::max(slots,graph.source_counts.at(i));
  return slots;
}
namespace {
uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}
struct Footprint {int64_t slots;long double fixed,row;bool enabled;};
Footprint footprint(const ContentProfile& p,int64_t rows) {
  int64_t slots=1;bool enabled=false;
  for(size_t i=0;i<p.graph.nodes.size();++i)if(aggregate_kind(p.graph.nodes[i].aggregation)) {
    slots=std::max(slots,p.graph.source_counts[i]);enabled=true;
  }
  return {slots,16.L*(p.graph.nodes.size()+1.L)*(slots+4.L)+32.L*(rows+1.L),
    64.L*slots+128.L,enabled};
}
}
long double PackedAggregate::minimum_bytes(const ContentProfile& p,int64_t rows) {
  const auto f=footprint(p,rows);return f.enabled?f.fixed+f.row:0;
}
PackedAggregate::PackedAggregate(const ContentProfile& p,at::Device device,int64_t rows,int64_t chunk,int64_t budget)
    :rows_(rows),sources_(p.sources) {
  const auto f=footprint(p,rows);slots_=f.slots;
  if(!f.enabled||rows<1||chunk<1||budget<f.fixed+f.row)throw std::invalid_argument("device Aggregate minimum exceeds budget");
  chunk_=std::min({rows,chunk,int64_t((budget-f.fixed)/f.row)});
  reserved_=int64_t(f.fixed+chunk_*f.row);
  auto weights=at::zeros({int64_t(p.graph.nodes.size()),slots_},at::kFloat);
  std::vector<int64_t> kinds,lengths;
  for(size_t n=0;n<p.graph.nodes.size();++n) {
    const auto k=aggregate_kind(p.graph.nodes[n].aggregation),count=p.graph.source_counts[n];
    kinds.push_back(k);lengths.push_back(count);
    if(k>=2)for(int64_t slot=0;slot<count;++slot) {
      const auto prefix=k==2?"agg_mass_":"agg_logit_";
      const auto& value=p.model.nodes[n].extra.at(prefix+std::to_string(slot));
      if(value.dim()!=0||value.scalar_type()!=at::kFloat||!value.device().is_cpu()
          ||!at::isfinite(value).item<bool>())throw std::invalid_argument("invalid device Aggregate slot parameter");
      weights[n][slot].copy_(value);
    }
  }
  kinds_=at::tensor(kinds,at::kLong).to(device);lengths_=at::tensor(lengths,at::kLong).to(device);
  weights_=weights.to(device);chunks_=at::zeros({1},kinds_.options());
}
void PackedAggregate::append(CannProgram& p,const ReadyBatch& ready,const PackedSum& sum,const at::Tensor& error,bool vectorized) const {
  const auto chunk=chunk_,slots=slots_;
  const auto kinds=kinds_,lengths=lengths_,weights=weights_,sources=sources_,chunks=chunks_;
  auto coefficient=at::ones({rows_},weights.options()),ids=at::empty({chunk},kinds.options());
  auto cursor=at::zeros({1},kinds.options()),zero=at::zeros_like(cursor),branch=at::zeros_like(error);
  auto logits=at::empty({chunk,slots},weights.options()),prob=at::empty_like(logits);
  auto masses=at::empty_like(logits),total=at::empty({chunk,1},weights.options());
  auto plan=[&](int64_t mode,int64_t target) {
    p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_aggregate_plan)(1,stream,
      ptr(ready.fibers),ptr(ready.fiber_offsets),ptr(ready.counts),ptr(sum.keys),ptr(sources),ptr(kinds),ptr(lengths),ptr(weights),
      ptr(ids),ptr(cursor),ptr(branch),ptr(chunks),ptr(logits),ptr(prob),ptr(total),ptr(coefficient),ptr(error),slots,chunk,mode,target),
      "plan normalized Aggregate domains");},
      {ready.fibers,ready.fiber_offsets,ready.counts,sum.keys,sources,kinds,lengths,weights,ids,cursor,branch,chunks,logits,prob,total,coefficient,error});
  };
  plan(0,0);
  for(int64_t target:{2,3,4}) {
    auto begin=p.label(),body=p.label(),done=p.label();p.copy(cursor,zero);p.mark(begin);plan(1,target);
    p.branch(branch,{done,body});p.mark(body);
    if(target==2){p.softplus(logits,masses);p.sum(masses,1,true,total);p.divide(masses,total,prob);}
    else p.softmax(logits,1,prob);
    plan(2,target);p.branch(branch,{begin});p.mark(done);
  }
  const auto width=sum.content.size(1);const auto blocks=uint32_t(vectorized?std::min<int64_t>(32,rows_):1);
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_aggregate_apply)(blocks,stream,
    ptr(ready.fibers),ptr(ready.fiber_offsets),ptr(ready.counts),ptr(sum.order),ptr(kinds),ptr(coefficient),
    ptr(sum.weighted),ptr(sum.content),ptr(error),width,int64_t(vectorized)),"apply normalized Aggregate contributions");},
    {ready.fibers,ready.fiber_offsets,ready.counts,sum.order,kinds,coefficient,sum.weighted,sum.content,error});
}
} // namespace tide::device_online
