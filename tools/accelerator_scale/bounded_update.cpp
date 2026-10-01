#include "bounded.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace accelerator_scale::bounded {
Tensor masked_rows(const Tensor& value,const Tensor& mask) {
  auto m=mask;while(m.dim()<value.dim())m=m.unsqueeze(-1);
  return at::where(m,value,at::zeros_like(value));
}
Tensor choose_dependencies(const Tensor& mask,const Tensor& yes,const Tensor& no) {
  return at::where(mask.unsqueeze(-1),yes,no);
}
Event Program::update(Index node,Index time,const State& old,std::vector<Message> fiber) const {
  Event e;e.node=node;e.time=time;e.old=old;
  const auto& w=f_.model.nodes[node];const auto& spec=f_.graph.nodes[node];const auto d=w.bias.device();
  std::sort(fiber.begin(),fiber.end(),[](const auto& a,const auto& b){return std::tie(a.kind,a.source,a.position)<std::tie(b.kind,b.source,b.position);});
  Tensor present=at::zeros({limits_.batch},w.bias.options().dtype(at::kBool));
  auto incoming=empty_dependencies(d);std::vector<Tensor> terms,values;
  Tensor coefficients;
  if(spec.memory=="lh-add-repeat-v1") {
    std::vector<Tensor> logits;
    for(Index i=0;i<f_.graph.source_counts[node];++i)logits.push_back(w.extra.at("agg_logit_"+std::to_string(i)));
    coefficients=at::softmax(at::stack(logits).unsqueeze(0).expand({limits_.batch,-1}),1);
  }
  for(auto& message:fiber) {
    message.value=copy(message.value,d);message.present=move(message.present,d);
    present=present|message.present;
    incoming=incoming|(message.value.dependencies & message.present.unsqueeze(1));
    auto value=masked_rows(message.value.data,message.present);
    values.push_back(value);
    auto contribution=coefficients.defined()?value*coefficients.select(1,message.slot).unsqueeze(-1):value;
    auto dep=message.value.dependencies;
    if(coefficients.defined())dep=dep|dependency(w.extra.at("agg_logit_0"),d);
    e.contributions.push_back({contribution,dep & message.present.unsqueeze(1)});
    terms.push_back(contribution);
  }
  e.fiber=std::move(fiber);e.candidate=present;
  auto content=terms[0];for(size_t i=1;i<terms.size();++i)content=content+terms[i];
  auto content_dep=incoming;
  if(coefficients.defined())content_dep=content_dep|dependency(w.extra.at("agg_logit_0"),d);
  e.content={content,content_dep & present.unsqueeze(1)};
  const auto& clock=spec.state_clock;
  auto old_local=at::where(old.last<0,old.last,at::floor_divide(old.last,clock.period)*clock.count+old.last.remainder(clock.period)-clock.first);
  const auto local_time=clock.to_local(time);auto gap=local_time-old_local;
  State proposal=old;
  proposal.last=at::where(present,at::full_like(old.last,time),old.last);
  proposal.observations=old.observations+present.to(at::kLong);proposal.seen=old.seen|present;
  if(spec.identity) {
    // Identity boundaries emit content but own no evolving numeric state.
    // The public identity StateKernel returns the old state unchanged.
    proposal=old;proposal.seen=old.seen|present;
  } else if(coefficients.defined()) {
    auto decayed=old.value.data;
    for(Index k=0;k<=local_time;++k)
      decayed=at::where((gap>k).unsqueeze(-1),decayed*w.extra.at("add_retention"),decayed);
    proposal.value={content+decayed,old.value.dependencies|e.content.dependencies};
  } else {
    // Physical padding is excluded by separate presence masks. Queries see all
    // current-fiber rows and the complete visible old cache, as in the oracle.
    std::vector<size_t> order(e.fiber.size());
    for(size_t i=0;i<order.size();++i)order[i]=i;
    std::sort(order.begin(),order.end(),[&](auto a,auto b){return e.fiber[a].slot<e.fiber[b].slot;});
    std::vector<Tensor> rows,masks,weights;
    const auto pool=at::softmax(w.extra.at("fiber_pool").unsqueeze(0).expand({limits_.batch,-1}),1);
    for(auto i:order){rows.push_back(values[i]);masks.push_back(e.fiber[i].present);weights.push_back(pool.select(1,e.fiber[i].slot));}
    auto x=at::stack(rows,1);auto valid=at::stack(masks,1);const Index width=f_.model.width(),head=width/4;
    auto qkv=at::linear(x,w.extra.at("fiber_qkv").t(),w.extra.at("fiber_qkv_bias")).split(width,-1);
    auto q=qkv[0].reshape({limits_.batch,Index(rows.size()),4,head}).transpose(1,2)*(1/std::sqrt(double(head)));
    auto key=at::cat({old.key,qkv[1].reshape({limits_.batch,Index(rows.size()),4,head})},1);
    auto value=at::cat({old.cache_value,qkv[2].reshape({limits_.batch,Index(rows.size()),4,head})},1);
    auto bias=old.bias;
    for(Index k=0;k<=local_time;++k)
      bias=at::where((present & (gap>k)).unsqueeze(-1),bias-w.extra.at("fiber_decay"),bias);
    bias=at::cat({bias,at::zeros({limits_.batch,Index(rows.size())},x.options())},1);
    auto cache_valid=at::cat({old.valid,valid},1);
    auto scores=at::matmul(key.transpose(1,2),q.transpose(2,3)).transpose(2,3)+bias.unsqueeze(1).unsqueeze(1);
    scores=scores.masked_fill(~cache_valid.unsqueeze(1).unsqueeze(1),-std::numeric_limits<float>::infinity());
    // Empty rows never create candidates; avoid all -inf softmax/NaN in their
    // predicated arithmetic. They contribute neither values nor dependencies.
    scores=at::where(cache_valid.any(1).reshape({limits_.batch,1,1,1}),scores,at::zeros_like(scores));
    auto output=at::matmul(at::softmax(scores,-1),value.transpose(1,2)).transpose(1,2).contiguous().reshape({limits_.batch,Index(rows.size()),width});
    output=masked_rows(output,valid);
    auto pooled=at::matmul(at::stack(weights,1).unsqueeze(1),output).squeeze(1);
    auto kv_dep=old.cache_dependencies|incoming|dependency(w.extra.at("fiber_qkv"),d);
    proposal.value={at::linear(pooled,w.extra.at("fiber_out").t(),w.extra.at("fiber_out_bias")),
      kv_dep|dependency(w.extra.at("fiber_pool"),d)|dependency(w.extra.at("fiber_out"),d)};
    proposal.key=key;proposal.cache_value=value;proposal.bias=bias;proposal.valid=cache_valid;
    proposal.cache_dependencies=choose_dependencies(present,kv_dep,old.cache_dependencies);
  }
  // A noncandidate preserves the stored state, including its dependency graph.
  proposal.value.data=at::where(present.unsqueeze(1),proposal.value.data,old.value.data);
  proposal.value.dependencies=choose_dependencies(present,proposal.value.dependencies,old.value.dependencies);
  e.proposal=proposal;
  if(spec.identity)
    e.descriptor={at::zeros({limits_.batch},w.bias.options()),empty_dependencies(d)};
  else if(spec.readout=="scale-norm-fp32-v1")
    e.descriptor={at::norm(proposal.value.data,2,{-1},false,at::kFloat),proposal.value.dependencies};
  else if(spec.readout=="linear-v1")
    e.descriptor={(proposal.value.data*w.read).sum(-1),proposal.value.dependencies};
  else throw std::invalid_argument("bounded Read profile requires explicit FP32");
  return e;
}
}  // namespace accelerator_scale::bounded
