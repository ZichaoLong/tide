#include "packed_fiber_pool.h"
#include "cann_api.h"
#include "aclrtlaunch_tide_fiber_chunk.h"
#include "aclrtlaunch_tide_fiber_pool.h"
#include <algorithm>

namespace tide::device_online {
namespace {
uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}
int64_t kind(const Node& node) {
  if(node.memory=="lh-fiber-attention-sum-repeat-v1")return 0;
  if(node.memory=="lh-fiber-attention-mean-repeat-v1")return 1;
  if(node.memory=="lh-fiber-attention-linear-repeat-v1")return 2;
  if(node.memory=="lh-fiber-attention-active-softmax-repeat-v1")return 3;
  if(node.memory=="lh-fiber-attention-all-softmax-repeat-v1")return 4;
  throw std::invalid_argument("unknown device fiber pooling profile");
}
std::pair<int64_t,int64_t> dimensions(const StateKernelProfile& p) {
  int64_t parameters=0,slots=1;
  for(size_t i=0;i<p.nodes.size();++i)if(!p.nodes[i].identity&&is_fiber_attention_profile(p.nodes[i].memory)) {
    ++parameters;slots=std::max(slots,p.source_counts[i]);
  }
  return {parameters,slots};
}
} // namespace
long double PackedFiberPool::reserved_bytes(const StateKernelProfile& p,int64_t rows,int64_t chunk) {
  const auto [parameters,slots]=dimensions(p);
  return 8.L*(parameters+1.L)*slots+64.L*(rows+parameters+chunk+1.L)+32.L*chunk*slots;
}
PackedFiberPool::PackedFiberPool(const StateKernelProfile& profile,at::Device device,int64_t rows,int64_t chunk)
    :rows_(rows),chunk_(chunk),inputs_(profile.input_count),sources_(profile.sources) {
  std::tie(parameters_,slots_)=dimensions(profile);
  auto weights=at::zeros({parameters_+1,slots_},at::kFloat);
  std::vector<int64_t> kinds,lengths;int64_t parameter=0;
  for(size_t n=0;n<profile.nodes.size();++n) {
    const auto& node=profile.nodes[n];if(node.identity||!is_fiber_attention_profile(node.memory))continue;
    const auto k=kind(node),length=profile.source_counts[n];kinds.push_back(k);lengths.push_back(length);
    if(k>=2)weights[parameter].narrow(0,0,length).copy_(profile.weights[n].extra.at("fiber_pool"));
    ++parameter;
  }
  kinds_=at::tensor(kinds,at::kLong).to(device);lengths_=at::tensor(lengths,at::kLong).to(device);weights_=weights.to(device);
}
at::Tensor PackedFiberPool::append(CannProgram& p,const at::Tensor& events,const at::Tensor& tokens,
    const at::Tensor& counts,const ReadyBatch& ready,const at::Tensor& error,const at::Tensor& chunks) const {
  const auto rows=rows_,chunk=chunk_,parameters=parameters_,slots=slots_,inputs=inputs_;
  const auto kinds=kinds_,lengths=lengths_,weights=weights_,sources=sources_;
  auto coefficient=at::ones({rows},weights.options());
  auto ids=at::empty({chunk},kinds.options()),source=at::empty_like(ids),parameter=at::empty_like(ids),destination=at::empty_like(ids);
  auto cursor=at::zeros({1},kinds.options()),zero=at::zeros_like(cursor),branch=at::zeros_like(error);
  auto logits=at::empty({chunk,slots},weights.options()),prob=at::empty_like(logits);
  auto task=[&](int64_t mode) {
    p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_fiber_pool)(1,stream,
      ptr(events),ptr(tokens),ptr(counts),ptr(ready.atoms.coordinates),ptr(sources),ptr(kinds),ptr(lengths),ptr(weights),
      ptr(ids),ptr(logits),ptr(prob),ptr(coefficient),ptr(error),rows,slots,chunk,inputs,mode),"post-attention pooling coefficients");},
      {events,tokens,counts,ready.atoms.coordinates,sources,kinds,lengths,weights,ids,logits,prob,coefficient,error});
  };
  task(0); // Linear coefficients only; sum/mean do not evaluate a softmax.
  for(int64_t target:{3,4}) {
    auto head=p.label(),body=p.label(),done=p.label();p.copy(cursor,zero);p.mark(head);
    p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_fiber_chunk)(1,stream,
      ptr(events),ptr(tokens),ptr(counts),ptr(kinds),ptr(cursor),ptr(source),ptr(parameter),ptr(destination),ptr(ids),
      ptr(branch),ptr(chunks),ptr(error),rows,parameters,chunk,int64_t(3),target),"pack actual softmax pooling events");},
      {events,tokens,counts,kinds,cursor,source,parameter,destination,ids,branch,chunks,error});
    p.branch(branch,{done,body});p.mark(body);task(1);p.softmax(logits,1,prob);task(2);p.branch(branch,{head});p.mark(done);
  }
  return coefficient;
}
} // namespace tide::device_online
