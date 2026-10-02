#include "sharded_state.h"
#include "cann_api.h"
#include "aclrtlaunch_tide_state_shard_merge.h"
#include <c10/core/impl/VirtualGuardImpl.h>
#include <algorithm>

namespace tide::device_online {
namespace {
uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}
std::vector<std::vector<int64_t>> owned_nodes(const FullPlacement& p,int64_t nodes,at::Device coordinator) {
  validate_full_placement(p,nodes,coordinator);std::vector<std::vector<int64_t>> out(p.devices.size());
  for(int64_t n=0;n<nodes;++n)out[p.owners[n]].push_back(n);return out;
}
long double common_bytes(const ContentProfile& p,const FullPlacement& placement,const ContentLimits& l) {
  return placement.devices.size()*(96.L*l.queue*(8.L*p.width+64)+16.L*p.graph.nodes.size()+4096);
}
void merge_error(CannProgram& p,const at::Tensor& from,const at::Tensor& to) {
  auto zero=at::zeros_like(from),ok=at::zeros({1},from.options().dtype(at::kBool)),branch=at::zeros_like(from);
  auto copy=p.label(),done=p.label();p.equal(from,zero,ok);p.cast_index(ok,branch);
  p.branch(branch,{copy,done});p.mark(copy);p.copy(to,from);p.mark(done);
}
}
struct ShardedState::Impl {
  struct Shard {
    std::unique_ptr<StateOwner> owner;
    std::unique_ptr<RemoteState> remote;
    at::Tensor mapping,error;
    StateShardBatch packed;
    StateReadProposal read;
    SelectionProposal selection;
    ContentUpdate update;
    RemoteStateResult result;
  };
  at::Device coordinator;ContentLimits limits;int64_t reserved=0;
  std::vector<Shard> shards;
  Impl(at::Device d,ContentLimits l):coordinator(d),limits(l) {}
};
long double ShardedState::minimum_bytes(const ContentProfile& p,const FullPlacement& placement,const Continuation& q,const ContentLimits& l) {
  if(placement.devices.empty())throw std::invalid_argument("state placement has no devices");
  long double out=common_bytes(p,placement,l);
  for(const auto& owned:owned_nodes(placement,p.graph.nodes.size(),placement.devices.front()))out+=StateOwner::minimum_bytes(p,owned,q,l);
  return out;
}
ShardedState::ShardedState(const ContentProfile& p,FullPlacement placement,const Continuation& q,at::Device d,ContentLimits l,int64_t budget)
    :impl_(std::make_unique<Impl>(d,l)) {
  const auto nodes=owned_nodes(placement,p.graph.nodes.size(),d);const auto minimum=minimum_bytes(p,placement,q,l);
  if(minimum>budget)throw std::invalid_argument("state shards exceed tensor budget");
  const auto extra=(budget-minimum)/nodes.size();impl_->reserved=int64_t(common_bytes(p,placement,l));
  for(size_t i=0;i<nodes.size();++i) {
    Impl::Shard s;s.owner=std::make_unique<StateOwner>(p,nodes[i],q,placement.devices[i],l,int64_t(StateOwner::minimum_bytes(p,nodes[i],q,l)+extra));
    impl_->reserved+=s.owner->reserved_bytes();std::vector<int64_t> map(p.graph.nodes.size(),-1);
    for(size_t j=0;j<nodes[i].size();++j)map[nodes[i][j]]=j;
    s.mapping=at::tensor(map,at::kLong).to(d);impl_->shards.push_back(std::move(s));
  }
  if(impl_->reserved>budget)throw std::logic_error("state shards exceeded admission");
}
ShardedState::~ShardedState()=default;
std::vector<StateOwnerValues> ShardedState::state_values() const {
  std::vector<StateOwnerValues> out;
  for(const auto& s:impl_->shards)out.push_back({s.owner->global_nodes(),s.owner->state().values,s.owner->state().present});
  return out;
}
std::vector<ContinuationRows> ShardedState::continuation_rows() const {
  std::vector<ContinuationRows> out;
  for(const auto& s:impl_->shards){auto xs=s.owner->continuation_rows();out.insert(out.end(),xs.begin(),xs.end());}
  return out;
}
std::vector<Tensor> ShardedState::continuation_tensors() const {
  std::vector<Tensor> out;
  for(const auto& s:impl_->shards){auto xs=s.owner->continuation_tensors();out.insert(out.end(),xs.begin(),xs.end());}
  return out;
}
void ShardedState::append_read(CannProgram& p,const ReadyBatch& ready,const ContentBatch& content,const at::Tensor& stage,
    const at::Tensor& error,int64_t operator_budget) {
  const auto rows=ready.fibers.size(0);auto scores=at::zeros({rows*2},content.scores.options());p.zero(scores);
  for(auto& s:impl_->shards) {
    s.error=at::zeros_like(error);p.copy(s.error,error);
    s.packed=append_state_shard_pack(p,ready,content,s.mapping,s.owner->global_nodes().size(),rows,s.error);
  }
  for(auto& s:impl_->shards)if(s.owner->state().values.device()!=impl_->coordinator) {
    s.remote=std::make_unique<RemoteState>(*s.owner,operator_budget);
    s.result=s.remote->append_read_send(p,s.packed,stage,s.error);
  }
  for(auto& s:impl_->shards)if(!s.remote) {
    s.read=s.owner->append_read(p,s.packed.ready,s.packed.content,s.error);
    s.result={s.packed.content.scores,{},{},{},s.error};
  }
  for(auto& s:impl_->shards) {
    if(s.remote)s.remote->append_read_receive(p);merge_error(p,s.result.error,error);
    append_state_shard_scatter(p,s.packed,s.result.scores,scores);
  }
  p.copy(content.scores,scores.narrow(0,0,rows));
}
ContentUpdate ShardedState::append_update(CannProgram& p,const ReadyBatch& ready,const ContentBatch& content,
    const SelectionProposal& selection,const at::Tensor& stage,const at::Tensor& event_count,const at::Tensor& error) {
  const auto rows=ready.fibers.size(0),width=content.content.size(1);const int64_t diagnostics=impl_->limits.diagnostics;
  auto comparison=at::zeros({rows*2,width},content.content.options());
  auto values=at::zeros({diagnostics?rows*2:1,diagnostics?5*width+2:1},content.scores.options());
  ContentUpdate out{{},{at::zeros({rows,4},ready.fibers.options()),content.content,selection.active},comparison.narrow(0,0,rows),
    at::zeros({diagnostics?rows:1,13},ready.fibers.options()),diagnostics?values.narrow(0,0,rows):values};
  for(auto& s:impl_->shards) {
    s.selection=append_state_shard_selection(p,s.packed,selection,error);
    if(s.remote)s.remote->append_update_send(p,s.selection,error);
  }
  for(auto& s:impl_->shards)if(!s.remote) {
    p.copy(s.error,error);s.update=s.owner->append_update(p,s.packed.ready,s.packed.content,s.selection,s.read,stage,s.error);
    s.result.comparison=s.update.comparison;s.result.event_meta=s.update.event_meta;s.result.event_values=s.update.event_values;
  }
  for(auto& s:impl_->shards) {
    if(s.remote)s.remote->append_update_receive(p);merge_error(p,s.result.error,error);
    append_state_shard_scatter(p,s.packed,s.result.comparison,comparison);
    if(diagnostics)append_state_shard_scatter(p,s.packed,s.result.event_values,values);
    const auto packed=s.packed;const auto result=s.result;
    p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_state_shard_merge)(1,stream,
      ptr(ready.fibers),ptr(ready.counts),ptr(packed.fiber_rows),ptr(packed.ready.counts),ptr(result.event_meta),
      ptr(out.event_meta),ptr(out.actions.coordinates),ptr(event_count),ptr(error),rows,diagnostics,int64_t(0)),"restore global state event identities");},
      {ready.fibers,ready.counts,packed.fiber_rows,packed.ready.counts,result.event_meta,out.event_meta,out.actions.coordinates,event_count,error});
  }
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_state_shard_merge)(1,stream,
    ptr(ready.fibers),ptr(ready.counts),ptr(ready.counts),ptr(ready.counts),ptr(out.event_meta),ptr(out.event_meta),
    ptr(out.actions.coordinates),ptr(event_count),ptr(error),rows,diagnostics,int64_t(1)),"count complete global state stage");},
    {ready.fibers,ready.counts,out.event_meta,out.actions.coordinates,event_count,error});
  return out;
}
void ShardedState::append_commit(CannProgram& p,const at::Tensor& error) {
  for(auto& s:impl_->shards)if(s.remote)s.remote->append_commit(p,error);
    else s.owner->append_commit(p,s.update,s.read,s.selection,error);
}
void ShardedState::append_stop(CannProgram& p){for(auto& s:impl_->shards)if(s.remote)s.remote->append_stop(p);}
void ShardedState::reset_window(){for(auto& s:impl_->shards)s.owner->reset_window();}
void ShardedState::synchronize_inputs() const {for(const auto& s:impl_->shards){auto d=s.owner->state().values.device();c10::impl::VirtualGuardImpl(d.type()).synchronizeDevice(d.index());}}
void ShardedState::submit(){for(auto& s:impl_->shards)if(s.remote)s.remote->submit();}
void ShardedState::wait(){std::exception_ptr e;for(auto& s:impl_->shards)if(s.remote)try{s.remote->wait();}catch(...){if(!e)e=std::current_exception();}if(e)std::rethrow_exception(e);}
void ShardedState::close(){for(auto& s:impl_->shards)if(s.remote)s.remote->close();}
void ShardedState::export_states(Continuation& q) const{for(const auto& s:impl_->shards)s.owner->export_states(q);}
void ShardedState::export_trace(std::vector<Event>& e) const{for(const auto& s:impl_->shards)s.owner->export_trace(e);}
int64_t ShardedState::reserved_bytes() const{return impl_->reserved;}
int64_t ShardedState::program_count() const{int64_t n=1;for(const auto& s:impl_->shards)n+=s.owner->state().values.device()!=impl_->coordinator;return n;}
int64_t ShardedState::workspace_bytes() const{int64_t n=0;for(const auto& s:impl_->shards)if(s.remote)n+=s.remote->workspace_bytes();return n;}
int64_t ShardedState::packet_bytes() const{int64_t n=0;for(const auto& s:impl_->shards)if(s.remote)n+=s.remote->packet_bytes();return n;}
std::map<std::string,int64_t> ShardedState::stats() const {
  std::map<std::string,int64_t> out{{"state_shards",int64_t(impl_->shards.size())},{"state_shard_nodes",0},{"state_shard_max_nodes",0}};
  for(size_t i=0;i<impl_->shards.size();++i) {
    const auto n=int64_t(impl_->shards[i].owner->global_nodes().size());out["state_shard_nodes"]+=n;
    out["state_shard_max_nodes"]=std::max(out["state_shard_max_nodes"],n);out["state_shard_"+std::to_string(i)+"_nodes"]=n;
    for(const auto& [name,value]:impl_->shards[i].owner->stats()) {
      if(name.find("_peak")!=std::string::npos||name.find("_capacity")!=std::string::npos||name.find("_rows")!=std::string::npos)
        out[name]=std::max(out[name],value);
      else out[name]+=value;
    }
    if(impl_->shards[i].remote)out["state_peer_retained_tensor_bytes"]+=impl_->shards[i].remote->retained_tensor_bytes();
  }
  out["state_peer_packet_bytes"]=packet_bytes();return out;
}
std::vector<StateOwnerTape> ShardedState::reverse_parameters(int64_t budget) const {
  long double bytes=0;for(const auto& s:impl_->shards)bytes+=s.owner->reverse_parameter_bytes();
  if(budget<1||2.L*bytes>budget)throw std::invalid_argument("compact owner tape gather/retention exceeds total tensor budget");
  std::vector<StateOwnerTape> out;for(const auto& s:impl_->shards)out.push_back(s.owner->reverse_parameters(budget));return out;
}
std::vector<StateOwnerBanks> ShardedState::parameter_banks() const {
  std::vector<StateOwnerBanks> out;for(const auto& s:impl_->shards)out.push_back(s.owner->parameter_banks());return out;
}
} // namespace tide::device_online
