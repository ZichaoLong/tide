#pragma once
#include "sharded_full_vjp.h"
#include "parameter_plan.h"
#include "full_vjp_fixture.h"
namespace tide::device_online::test {
// Assertion boundary only. Download completed partials and sum aliases on CPU
// for comparison. These tensors are never inputs to a candidate or optimizer.
inline std::map<std::string,Tensor> sharded_owner_observations(const Graph& graph,const ParameterRegistry& registry,
                                                            const std::vector<ShardedGraphVjp>& gradients) {
  std::map<std::string,Tensor> result;
  for(const auto& owner:registry.owners())result.emplace(owner.canonical,Tensor{});
  for(const auto& out:gradients) {
    const auto& g=out.coordinator;const int64_t nodes=graph.nodes.size(),width=g.decay.size(1);
    const auto plan=plan_parameters(graph,registry,width,1024*1024*1024,g.read.defined());
    std::vector<Tensor> bank(14),on(14);auto cpu=[](const Tensor& x){return x.defined()?x.cpu():Tensor{};};
    if(plan.has_tanh){bank[0]=at::zeros({nodes,width,width},at::kFloat);bank[1]=at::zeros({nodes,width},at::kFloat);}
    if(plan.has_lh){bank[5]=at::zeros({nodes,width},at::kFloat);bank[6]=at::zeros_like(bank[5]);}
    if(plan.swiglu_count){bank[7]=at::zeros({plan.swiglu_count,width,2*width},at::kFloat);bank[8]=at::zeros_like(bank[7]);bank[9]=at::zeros({plan.swiglu_count,2*width,width},at::kFloat);}
    std::vector<int64_t> swiglu(nodes,-1);int64_t k=0;for(int64_t n=0;n<nodes;++n)if(!graph.nodes[n].identity&&graph.nodes[n].full=="swiglu")swiglu[n]=k++;
    auto connections=at::zeros({nodes},at::kBool);
    for(const auto& shard:out.full->gradients()) {
      auto ids=at::tensor(shard.nodes,at::kLong);const auto& f=shard.values;
      connections.index_copy_(0,ids,f.parameter_connected.cpu());
      for(const auto& pair:{std::make_pair(0,f.weights),std::make_pair(1,f.biases),std::make_pair(5,f.extra.lh_weights),std::make_pair(6,f.extra.lh_biases)})
        if(pair.second.defined())bank[pair.first].index_copy_(0,ids,pair.second.cpu());
      std::vector<int64_t> compact;for(auto n:shard.nodes)if(swiglu[n]>=0)compact.push_back(swiglu[n]);
      if(!compact.empty())for(const auto& pair:{std::make_pair(7,f.extra.gate),std::make_pair(8,f.extra.up),std::make_pair(9,f.extra.down)})
        bank[pair.first].index_copy_(0,at::tensor(compact,at::kLong),pair.second.cpu());
    }
    if(!at::equal(connections,g.full_connected.cpu()))throw std::runtime_error("shard/global Full connection disagreement");
    bank[2]=cpu(g.decay);bank[3]=cpu(g.retention);bank[4]=cpu(g.scales);bank[10]=cpu(g.aggregate.values);
    bank[11]=cpu(g.read);bank[12]=cpu(g.attention);bank[13]=cpu(g.fiber);
    for(int i:{0,1,5,6,7,8,9})on[i]=connections;
    on[2]=cpu(g.decay_connected);on[3]=cpu(g.retention_connected);on[4]=cpu(g.scale_connected);on[10]=cpu(g.aggregate.connected);
    on[11]=cpu(g.read_connected);on[12]=cpu(g.attention_connected);on[13]=cpu(g.fiber_connected);
    for(auto& x:bank)if(x.defined())x=x.reshape({-1});for(auto& x:on)if(x.defined())x=x.reshape({-1});
    for(size_t owner=0;owner<plan.owners.size();++owner) {
      const auto& o=plan.owners[owner];auto& sum=result.at(o.canonical);
      const auto first=plan.owner_table[owner*4],last=plan.owner_table[owner*4+1],size=o.value.numel();
      for(int64_t ref=first;ref<last;++ref) {
        const auto b=plan.references[ref*3],c=plan.references[ref*3+1],offset=plan.references[ref*3+2];
        if(!on[b][c].item<bool>())continue;
        auto part=bank[b].narrow(0,offset,size).reshape(o.value.sizes());if(!sum.defined())sum=part.clone();else sum.add_(part);
      }
    }
  }
  return result;
}
} // namespace tide::device_online::test
