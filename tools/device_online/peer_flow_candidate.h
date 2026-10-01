#pragma once
#include "content_flow.h"

namespace tide::device_online::test {
// Test client only: the same fixtures/assertions exercise the unchanged public
// single-device owner or the explicit internal peer Full inference increment.
class InferenceCandidate {
 public:
  InferenceCandidate(const Graph& g,const Model& m,const Continuation& q,at::Device d,ResidentLimits l,bool peer) {
    if(peer)remote_=std::make_unique<ContentFlow>(g,m,q,d,l,at::Device(d.type(),d.index()+1));
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
    if(r.stats.at("full_peer_devices")!=(remote_?2:1))throw std::runtime_error("wrong effective Full placement");
    if(r.stats.at("planned_headroom_bytes")<0)throw std::runtime_error("peer exceeded total declared workspace budget");
    return r;
  }
  Continuation snapshot() const{return local_?local_->snapshot():remote_->snapshot();}
  void close(){if(local_)local_->close();else remote_->close();}
 private:
  std::unique_ptr<ResidentSession> local_;
  std::unique_ptr<ContentFlow> remote_;
};
} // namespace tide::device_online::test
