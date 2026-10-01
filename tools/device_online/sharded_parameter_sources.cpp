#include "sharded_parameter_sources.h"
#include "sharded_state_vjp.h"
#include "parameter_plan.h"
#include <algorithm>
#include <numeric>
#include <limits>
#include <stdexcept>
#include <set>

namespace tide::device_online {
namespace {
ParameterContribution slice(const at::Tensor& values,const at::Tensor& connected,
                            int64_t offset,int64_t size,int64_t connection) {
  if(!values.defined()||!connected.defined()||values.device().type()!=c10::DeviceType::PrivateUse1
      ||values.device()!=connected.device()||values.scalar_type()!=at::kFloat||connected.scalar_type()!=at::kBool
      ||!values.is_contiguous()||!connected.is_contiguous()||values.requires_grad()||connected.requires_grad()
      ||offset<0||size<1||size>values.numel()||offset>values.numel()-size||connection<0||connection>=connected.numel())
    throw std::invalid_argument("invalid device parameter contribution");
  return {values.reshape({-1}).narrow(0,offset,size),connected.reshape({-1}).narrow(0,connection,1)};
}
struct FullNode {size_t shard;int64_t local,swiglu;};
std::vector<FullNode> full_nodes(const Graph& graph,const std::vector<FullShardGradient>& shards) {
  std::vector<FullNode> out(graph.nodes.size());std::vector<bool> seen(out.size(),false);
  for(size_t s=0;s<shards.size();++s) {
    int64_t swiglu=0;
    for(size_t local=0;local<shards[s].nodes.size();++local) {
      const auto n=shards[s].nodes[local];
      if(n<0||n>=int64_t(out.size())||seen[n])throw std::invalid_argument("invalid Full partial node map");
      seen[n]=true;const bool ffn=!graph.nodes[n].identity&&graph.nodes[n].full=="swiglu";
      out[n]={s,int64_t(local),ffn?swiglu++:-1};
    }
  }
  if(std::find(seen.begin(),seen.end(),false)!=seen.end())throw std::invalid_argument("incomplete Full partial node map");
  return out;
}
}
ShardedParameterSources sharded_parameter_sources(const Graph& graph,const ParameterRegistry& registry,
    const std::vector<ShardedGraphVjp>& windows,int64_t budget) {
  if(windows.empty()||graph.nodes.empty()||budget<1)throw std::invalid_argument("parameter reduction requires retained graph partials");
  ShardedParameterSources out;out.owners=registry.owners();out.contributions.resize(out.owners.size());
  long double used=128.L*out.owners.size();
  if(used>budget)throw std::invalid_argument("parameter owner metadata budget exceeded");
  // This order matches the existing retained reverse accumulator, including
  // aliases whose contributions live in different parameter banks/devices.
  for(size_t w=windows.size();w>0;) {--w;
    const auto& window=windows[w];const auto& g=window.coordinator;
    if(!window.full||!g.decay.defined()||g.decay.dim()!=2)throw std::invalid_argument("incomplete sharded graph gradients");
    // plan_parameters also estimates a dense output, which is never allocated
    // here. Apply this planner's metadata bound below; each device reduction
    // separately admits its actual partition's numerical storage.
    const auto width=g.decay.size(1);const auto plan=plan_parameters(graph,registry,width,std::numeric_limits<int64_t>::max(),g.read.defined());
    const auto shards=window.full->gradients();const auto mapping=full_nodes(graph,shards);
    const auto states=window.state?window.state->gradients():std::vector<StateShardGradient>{};
    std::vector<std::pair<size_t,int64_t>> state_map(graph.nodes.size(),{0,-1});
    for(size_t s=0;s<states.size();++s)for(size_t n=0;n<states[s].nodes.size();++n) {
      const auto global=states[s].nodes[n];
      if(global<0||global>=int64_t(state_map.size())||state_map[global].second!=-1)throw std::invalid_argument("invalid state contribution map");
      state_map[global]={s,int64_t(n)};
    }
    if(!states.empty())for(auto entry:state_map)if(entry.second<0)throw std::invalid_argument("incomplete state contributions");
    std::vector<std::pair<size_t,int64_t>> projection_map;
    if(!g.emission.shards.empty()) {
      projection_map.resize(g.emission.connected.numel(),{0,-1});
      for(size_t s=0;s<g.emission.shards.size();++s)for(size_t i=0;i<g.emission.shards[s].rows.size();++i) {
        const auto row=g.emission.shards[s].rows[i];
        if(row<0||row>=int64_t(projection_map.size())||projection_map[row].second>=0)
          throw std::invalid_argument("invalid compact projection contribution map");
        projection_map[row]={s,int64_t(i)};
      }
      for(auto entry:projection_map)if(entry.second<0)throw std::invalid_argument("incomplete projection contributions");
    }
    const auto event_offsets=event_parameter_offsets(graph,width),fiber_offsets=fiber_parameter_offsets(graph,width);
    used+=64.L*plan.references.size();if(used>budget)throw std::invalid_argument("parameter contribution metadata budget exceeded");
    for(size_t i=0;i<out.owners.size();++i) {
      const auto first=plan.owner_table[i*4],last=plan.owner_table[i*4+1],size=out.owners[i].value.numel();
      for(int64_t r=first;r<last;++r) {
        const auto bank=plan.references[r*3],connection=plan.references[r*3+1],offset=plan.references[r*3+2];
        ParameterContribution part;
        if(bank<2||(bank>=5&&bank<=9)) {
          if(connection<0||connection>=int64_t(mapping.size()))throw std::invalid_argument("Full parameter node is outside graph");
          const auto m=mapping[connection];const auto& f=shards[m.shard].values;
          const auto values=bank==0?f.weights:bank==1?f.biases:bank==5?f.extra.lh_weights:bank==6?f.extra.lh_biases:
            bank==7?f.extra.gate:bank==8?f.extra.up:f.extra.down;
          const int64_t local=bank>=7?m.swiglu:m.local;
          if(local<0)throw std::invalid_argument("SwiGLU contribution has no local owner");
          part=slice(values,f.parameter_connected,local*size,size,m.local);
        } else if(!states.empty()&&(bank==2||bank==3||bank==11||bank==12||bank==13)) {
          const int64_t stride=bank==12?4:bank==13?6:1,node=connection/stride;
          if(node<0||node>=int64_t(state_map.size()))throw std::invalid_argument("state contribution node outside graph");
          const auto [owner,local]=state_map[node];const auto& state=states[owner];
          const auto values=bank==2?state.decay:bank==3?state.retention:bank==11?state.read:bank==12?state.attention:state.fiber;
          const auto flags=bank==2?state.decay_connected:bank==3?state.retention_connected:bank==11?state.read_connected:
            bank==12?state.attention_connected:state.fiber_connected;
          const int64_t local_offset=bank==12?offset-event_offsets[node]+state.layout.event_offsets[local]:
            bank==13?offset-fiber_offsets[node]+state.layout.fiber_offsets[local]:local*(bank==3?1:width);
          part=slice(values,flags,local_offset,size,local*stride+connection%stride);
        } else if(!projection_map.empty()&&(bank==14||bank==15)) {
          const auto [owner,local]=projection_map.at(connection);const auto& projection=g.emission.shards[owner];
          part=slice(bank==14?projection.weights:projection.biases,projection.connected,local*size,size,local);
        } else {
          const auto values=bank==2?g.decay:bank==3?g.retention:bank==4?g.scales:bank==10?g.aggregate.values:
            bank==11?g.read:bank==12?g.attention:bank==13?g.fiber:bank==14?g.emission.weights:g.emission.biases;
          const auto on=bank==2?g.decay_connected:bank==3?g.retention_connected:bank==4?g.scale_connected:
            bank==10?g.aggregate.connected:bank==11?g.read_connected:bank==12?g.attention_connected:bank==13?g.fiber_connected:g.emission.connected;
          part=slice(values,on,offset,size,connection);
        }
        out.contributions[i].push_back(std::move(part));
      }
    }
  }
  return out;
}
std::vector<int64_t> place_parameter_owners(const ShardedParameterSources& p,int64_t devices) {
  std::vector<bool> active;for(const auto& parts:p.contributions)active.push_back(!parts.empty());
  return place_parameter_owners(p.owners,active,devices);
}
std::vector<int64_t> place_parameter_owners(const std::vector<ParameterOwner>& owners,const std::vector<bool>& active,int64_t devices) {
  if(devices<1||devices>16||owners.size()!=active.size())throw std::invalid_argument("invalid canonical owner placement");
  std::set<const void*> identities;std::set<std::string> names;
  for(const auto& owner:owners) {
    if(!owner.value.defined()||owner.value.numel()<1||!owner.value.device().is_cpu()
        ||(owner.value.scalar_type()!=at::kFloat&&owner.value.scalar_type()!=at::kHalf)
        ||owner.aliases.empty()||owner.canonical!=owner.aliases.front()
        ||!std::is_sorted(owner.aliases.begin(),owner.aliases.end())
        ||!identities.insert(owner.value.unsafeGetTensorImpl()).second)
      throw std::invalid_argument("invalid canonical parameter owner");
    for(const auto& name:owner.aliases)if(!names.insert(name).second)throw std::invalid_argument("duplicate parameter alias");
  }
  std::vector<size_t> order(owners.size());std::iota(order.begin(),order.end(),0);
  auto cost=[&](size_t i){return active[i]?owners[i].value.numel():0;};
  std::stable_sort(order.begin(),order.end(),[&](size_t a,size_t b){return cost(a)>cost(b);});
  std::vector<long double> load(devices,0);std::vector<int64_t> out(order.size());
  for(auto i:order) {
    const auto d=std::min_element(load.begin(),load.end())-load.begin();out[i]=d;load[d]+=4.L*cost(i);
  }
  return out;
}
} // namespace tide::device_online
