#include "weighted_flow_internal.h"
#include "tide/kernel.h"
#include <ATen/core/grad_mode.h>
namespace tide::device_online {
namespace {
WeightedRawLayout layout(const weighted::MemoryConfiguration& c,int64_t capacity){
  if(auto kv=std::get_if<weighted::KVConfig>(&c))return WeightedRawLayout(*kv,capacity);
  const auto& rc=std::get<weighted::RecurrentConfig>(c);return WeightedRawLayout(rc,rc.basic()?0:capacity);
}
State detach(State x){x.value=x.value.detach();for(auto& [_,v]:x.slots)v=v.detach();return x;}
}
WeightedFlowNode::WeightedFlowNode(const Model& model,int64_t node,const Continuation& q,at::Device d,int64_t capacity,int64_t budget,std::shared_ptr<ParameterBankPool> pool)
  :binding(weighted::node_binding(model.nodes.at(node))),layout(device_online::layout(binding.memory,capacity)),
   banks(model.nodes.at(node).extra,"nodes."+std::to_string(node)+".extra.",d,std::move(pool)){
  std::vector<State> initial_states,states;std::vector<int64_t> present;
  for(int64_t b=0;b<q.batch_size;++b){
    const auto& w=model.nodes.at(node);auto init=w.kernel->initial(w);initial_states.push_back(init);
    auto found=q.states.find({b,node});present.push_back(found!=q.states.end());states.push_back(found==q.states.end()?init:detach(found->second));
  }
  initial=copy_weighted_raw_state(encode_weighted_raw_state(layout,initial_states,budget),d);
  state=copy_weighted_raw_state(encode_weighted_raw_state(layout,states,budget),d);
  old={at::zeros_like(state.floating),at::zeros_like(state.integer)};
  state_present=at::tensor(present,at::kLong).to(d);
}
WeightedFlowRegion::WeightedFlowRegion(const Graph& g,const Model& m,int64_t region,const Continuation& q,at::Device d,std::shared_ptr<ParameterBankPool> pool)
  :config(weighted::region_binding(m.regions.at(region))),banks(m.regions.at(region).extra,"regions."+std::to_string(region)+".",d,std::move(pool)){
  for(auto i=g.region_index.offsets[region];i<g.region_index.offsets[region+1];++i)nodes.push_back(g.region_index.edges[i]);
  auto h=at::zeros({q.batch_size,2},at::kLong);h.select(1,0).fill_(-1);
  for(int64_t b=0;b<q.batch_size;++b){auto found=q.history.find({b,region});if(found!=q.history.end()){h[b][0]=found->second.last_time;h[b][1]=1;}}
  history=h.to(d);
}
Continuation WeightedFlow::Impl::checkpoint()const{
  if(failed)throw std::logic_error("failed weighted owner requires an earlier checkpoint and a new owner");
  auto out=boundary;
  for(size_t v=0;v<nodes.size();++v){const auto& n=*nodes[v];
    auto states=decode_weighted_raw_state(n.layout,copy_weighted_raw_state(n.state,at::kCPU));auto present=n.state_present.to(at::kCPU);
    for(int64_t b=0;b<out.batch_size;++b)if(present[b].item<int64_t>())out.states[{b,v}]=std::move(states[b]);
  }
  for(size_t r=0;r<regions.size();++r){auto h=regions[r]->history.to(at::kCPU);
    for(int64_t b=0;b<out.batch_size;++b)if(h[b][1].item<int64_t>())out.history[{b,r}]=History{h[b][0].item<int64_t>()};
  }
  const auto& q=pending->atoms();auto meta=q.coordinates.to(at::kCPU),values=q.values.to(at::kCPU),valid=q.valid.to(at::kCPU);
  for(int64_t i=0;i<valid.numel();++i)if(valid[i].item<bool>()){auto row=meta[i];out.pending.push_back({row[0].item<int64_t>(),row[1].item<int64_t>(),
    row[2].item<int64_t>(),row[3].item<int64_t>(),row[4].item<int64_t>(),row[5].item<int64_t>(),values[i].clone()});}
  std::sort(out.pending.begin(),out.pending.end(),[](const Atom& a,const Atom& b){return a.key()<b.key();});return out;
}
}
