#pragma once
#include "content_flow.h"
#include "sharded_state.h"

namespace tide::device_online::test {
// Test client only: the same fixtures/assertions exercise the unchanged public
// single-device owner or the explicit internal peer Full inference increment.
class InferenceCandidate {
 public:
  InferenceCandidate(const Graph& g,const Model& m,const Continuation& q,at::Device d,ResidentLimits l,bool peer,int shards=0,const std::string& policy="locality",bool state_shards=false) {
    devices_=shards?shards:peer?2:1;nodes_=g.nodes.size();shards_=shards;states_=state_shards;
    if(shards) {
      std::vector<at::Device> devices;for(int i=0;i<shards;++i)devices.emplace_back(d.type(),d.index()+i);
      auto full=place_full(g,m,std::move(devices),policy);
      if(state_shards) {
        auto state=full;for(auto& owner:state.owners)owner=(owner+1)%shards;
        remote_=std::make_unique<ContentFlow>(g,m,q,d,l,ModelPlacement{full,state});
      }else remote_=std::make_unique<ContentFlow>(g,m,q,d,l,std::move(full));
    }
    else if(peer)remote_=std::make_unique<ContentFlow>(g,m,q,d,l,at::Device(d.type(),d.index()+1));
    else local_=std::make_unique<ResidentSession>(g,m,q,d,l);
  }
  ResidentWindow advance(const std::vector<External>& input,Index stop,Index seal) {
    if(local_)return local_->advance(input,stop,seal);
    if(stop!=seal)throw std::invalid_argument("peer fixture requires a complete input seal");
    const auto w=remote_->advance_device(input,stop);
    return {w.outputs.coordinates,w.outputs.values,w.outputs.valid,w.output_stats,w.pending_stats,
            w.stages,w.events,w.full_chunks,w.emission_chunks};
  }
  Result result() const {
    auto r=local_?local_->result():remote_->result();
    if(r.stats.at("full_peer_devices")!=devices_)throw std::runtime_error("wrong effective Full placement");
    if(r.stats.at("planned_headroom_bytes")<0)throw std::runtime_error("peer exceeded total declared workspace budget");
    if(states_&&(r.stats.at("state_shard_nodes")!=nodes_||r.stats.at("state_shards")!=shards_))
      throw std::runtime_error("incomplete compact state placement");
    if(shards_) {
      int64_t n=0;for(int i=0;i<shards_;++i)n+=r.stats.at("full_shard_"+std::to_string(i)+"_nodes");
      if(n!=nodes_||r.stats.at("full_shards")!=shards_)throw std::runtime_error("incomplete Full shard assignment");
      if(r.stats.at("full_shard_capacity_rows")<r.stats.at("full_shard_selected_rows"))throw std::runtime_error("invalid shard packing counts");
      if(r.stats.at("diagnostics")) {
        int64_t selected=0;for(const auto& e:r.trace)selected+=e.active;
        if(selected!=r.stats.at("full_shard_selected_rows"))throw std::runtime_error("shard computed missing/extra actions");
      }
    }
    return r;
  }
  Continuation snapshot() const{return local_?local_->snapshot():remote_->snapshot();}
  void close(){if(local_)local_->close();else remote_->close();}
 private:
  int64_t devices_=1,nodes_=0,shards_=0;bool states_=false;
  std::unique_ptr<ResidentSession> local_;
  std::unique_ptr<ContentFlow> remote_;
};
} // namespace tide::device_online::test
