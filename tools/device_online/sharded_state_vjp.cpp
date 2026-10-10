#include "device_backend.h"
#include "sharded_state_vjp.h"
#include "state_reverse_merge.h"
#include "peer_exchange.h"
#include <c10/core/impl/VirtualGuardImpl.h>
#include <optional>
#include <set>
#include <stdexcept>
namespace tide::device_online {
struct ShardedStateVjp::Impl {
  struct Shard {
    explicit Shard(const StateOwnerTape& t):tape(t){}
    StateOwnerTape tape;
    StateReversePacket packed,local;
    StateReverseStage stage;
    StateOwnerVjp result;
    at::Tensor mapping,ids,error,remote_error,command,stop;
    std::vector<CacheCotangents> roots;
    std::vector<CacheGradient> next;
    std::optional<StateShardGradient> reuse;
    std::unique_ptr<StateOwnerReverse> reverse;
    std::unique_ptr<DeviceProgram> program;
    std::unique_ptr<PeerExchange> init,request,response,finish;
  };
  ReverseTape global;at::Tensor error;at::Device coordinator;
  int64_t chunk,budget,workspace;bool prepared=false,built=false;
  std::vector<Shard> shards;
  Impl(const ReverseTape& t):global(t),coordinator(t.state.metadata.device()){}
};
ShardedStateVjp::ShardedStateVjp(const ShardedReverseTape& t,const at::Tensor& error,int64_t chunk,int64_t budget,int64_t workspace,
    const std::shared_ptr<ShardedStateVjp>& next,const std::vector<std::vector<CacheCotangents>>& roots):impl_(std::make_unique<Impl>(t.coordinator)) {
  auto& s=*impl_;s.error=error;s.chunk=chunk;s.workspace=workspace;
  if(t.states.empty()||chunk<1||budget<1||workspace<1)throw std::invalid_argument("invalid compact state reverse ownership/budget");
  if(!roots.empty()&&roots.size()!=t.states.size())throw std::invalid_argument("state root owner count mismatch");
  s.budget=budget/int64_t(t.states.size());const int64_t nodes=t.coordinator.graph->nodes.size();
  const auto later=next?next->gradients():std::vector<StateShardGradient>{};
  if(next&&(later.size()!=t.states.size()||next->impl_->global.cut!=t.coordinator.stop
      ||next->impl_->global.graph->identity!=t.coordinator.graph->identity))throw std::invalid_argument("nonconsecutive compact state reverse bridge");
  std::set<int64_t> seen;std::set<c10::DeviceIndex> devices;
  for(size_t i=0;i<t.states.size();++i) {
    const auto& owner=t.states[i];const auto d=owner.state.decay.device();
    if(d.type()!=tide::device_online::resident_device_type||!devices.insert(d.index()).second||owner.layout.width!=s.global.full.width
        ||owner.state.samples!=s.global.state.samples||owner.global_nodes.size()!=size_t(owner.layout.nodes))
      throw std::invalid_argument("incompatible compact reverse owner");
    Impl::Shard shard(owner);if(!roots.empty())shard.roots=roots[i];std::vector<int64_t> mapping(nodes,-1);
    for(size_t j=0;j<owner.global_nodes.size();++j) {
      const auto n=owner.global_nodes[j];if(n<0||n>=nodes||!seen.insert(n).second)throw std::invalid_argument("duplicate compact reverse node");mapping[n]=j;
    }
    if(next) {
      const auto& old=next->impl_->shards[i].tape;
      if(owner.global_nodes!=later[i].nodes||d!=old.state.decay.device()||owner.attention.size()!=old.attention.size()||owner.fiber.size()!=old.fiber.size())
        throw std::invalid_argument("retained compact cache ownership changed without an explicit bridge");
      for(size_t j=0;j<owner.attention.size();++j)if(owner.attention[j].nodes!=old.attention[j].nodes)throw std::invalid_argument("event cache group changed");
      for(size_t j=0;j<owner.fiber.size();++j)if(owner.fiber[j].cache.nodes!=old.fiber[j].cache.nodes)throw std::invalid_argument("fiber cache group changed");
      shard.next=later[i].cache;
    }
    shard.mapping=at::tensor(mapping,at::kLong).to(s.coordinator);shard.ids=at::tensor(owner.global_nodes,at::kLong).to(s.coordinator);
    shard.error=at::zeros_like(error);
    if(d!=s.coordinator){shard.program=std::make_unique<DeviceProgram>(d);shard.program->limit_workspace(workspace);}
    s.shards.push_back(std::move(shard));
  }
  if(seen.size()!=size_t(nodes))throw std::invalid_argument("incomplete compact state reverse ownership");
}
ShardedStateVjp::~ShardedStateVjp()=default;
void ShardedStateVjp::reuse_attention_parameters(const ShardedStateVjp& next) {
  auto& s=*impl_;const auto& later=*next.impl_;
  if(s.prepared||s.built||later.global.cut!=s.global.stop||later.global.graph->identity!=s.global.graph->identity
      ||s.shards.size()!=later.shards.size())throw std::logic_error("attention adjoint reuse requires a following window before preparation");
  const auto gradients=next.gradients();
  for(size_t i=0;i<s.shards.size();++i) {
    auto& owner=s.shards[i];
    if(owner.reuse||owner.tape.global_nodes!=gradients[i].nodes||owner.tape.state.decay.device()!=later.shards[i].tape.state.decay.device())
      throw std::invalid_argument("attention adjoint reuse owner changed");
    owner.reuse=gradients[i];
  }
}
void ShardedStateVjp::prepare(DeviceProgram& p,const ReverseLinks& links) {
  auto& s=*impl_;if(s.prepared)throw std::logic_error("compact reverse preparation repeated");s.prepared=true;
  ReverseGatherInput events(p,s.global.state.values),fibers(p,s.global.fiber_values),scales(p,links.scales);
  for(auto& owner:s.shards) {
    owner.packed=append_state_reverse_pack(p,s.global,links,owner.mapping,owner.ids.numel(),s.global.state.metadata.size(0),
      s.global.fiber_values.size(0),s.error,s.budget/8,events,fibers,scales);owner.local=owner.packed;p.copy(owner.error,s.error);
    auto& program=owner.program?*owner.program:p;auto local_error=owner.program?owner.error:s.error;
    if(owner.program) {
      const auto d=owner.tape.state.decay.device();PeerExchange::Fields fields;
      auto copy=[&](at::Tensor& t){auto x=at::zeros(t.sizes(),t.options().device(d));fields.emplace_back(t,x);t=x;};
      auto& a=owner.local;for(auto* t:{&a.event_meta,&a.event_values,&a.event_count,&a.fiber_meta,&a.fiber_values,&a.fiber_count,
        &a.links.messages,&a.links.valid,&a.links.consumer_head,&a.links.consumer_next,&a.links.scales})copy(*t);
      copy(local_error);owner.init=std::make_unique<PeerExchange>(std::move(fields),s.budget/4);
      owner.init->append_send(p);owner.init->append_receive(program);
    }
    owner.reverse=std::make_unique<StateOwnerReverse>(program,owner.tape,owner.local,owner.roots,owner.next,local_error,s.chunk,s.budget/2,owner.reuse?&*owner.reuse:nullptr);
    // The preparation status lives on the owner and is included in the first
    // stage request/response. No host inspection controls the stage loop.
    owner.remote_error=local_error;
  }
}
StateVjp ShardedStateVjp::append_stage(DeviceProgram& p,const at::Tensor& range,const StateCotangents& cot,const ControlScores& scores) {
  auto& s=*impl_;if(!s.prepared||s.built)throw std::logic_error("compact reverse stage order invalid");s.built=true;
  const auto capacity=cot.events.size(0),width=cot.events.size(2);auto f=cot.events.options(),b=cot.connected.options();
  auto zero=[&](at::IntArrayRef shape,bool flag=false){auto x=at::empty(shape,flag?b:f);p.zero(x);return x;};
  StateVjp out{zero({capacity,width}),zero({capacity},true),zero(cot.final.sizes()),zero(cot.final_connected.sizes(),true),
    zero(cot.final.sizes()),zero(cot.final_connected.sizes(),true),zero(cot.final.sizes()),zero(cot.final_connected.sizes(),true)};
  ReverseGatherInput events(p,cot.events),connections(p,cot.connected);
  for(auto& owner:s.shards) {
    owner.stage=append_state_reverse_stage(p,owner.packed,owner.tape.state,range,cot,owner.ids,scores.scores,scores.read_connected,s.error,s.budget/8,events,connections);
    if(!owner.program){owner.result=owner.reverse->append_stage(p,owner.stage,s.error);continue;}
    auto& remote=*owner.program;const auto d=owner.tape.state.decay.device();auto stage=owner.stage;
    PeerExchange::Fields fields;auto copy=[&](at::Tensor& t){auto x=at::zeros(t.sizes(),t.options().device(d));fields.emplace_back(t,x);t=x;};
    owner.command=at::zeros_like(s.error);owner.stop=at::zeros_like(s.error);auto work=at::ones_like(s.error),command=owner.command;copy(command);
    for(auto* t:{&stage.tape.metadata,&stage.tape.values,&stage.tape.count,&stage.range,&stage.cot.events,&stage.cot.connected,
      &stage.cot.final,&stage.cot.final_connected})copy(*t);
    if(stage.score_gradient.defined()){copy(stage.score_gradient);copy(stage.read_connected);}
    auto incoming=s.error;copy(incoming);const auto local_error=owner.remote_error;
    owner.request=std::make_unique<PeerExchange>(std::move(fields),s.budget/4);
    const auto head=remote.label(),body=remote.label(),done=remote.label();auto again=at::zeros_like(command);
    remote.mark(head);owner.request->append_receive(remote);remote.branch(command,{done,body});remote.mark(body);
    // Preserve a preparation failure and propagate any coordinator failure.
    append_state_reverse_error(remote,incoming,local_error);
    auto local=owner.reverse->append_stage(remote,stage,local_error);owner.result=local;
    PeerExchange::Fields response;auto receive=[&](at::Tensor& t){auto x=at::zeros(t.sizes(),t.options().device(s.coordinator));response.emplace_back(t,x);t=x;};
    auto& r=owner.result;for(auto* t:{&r.state.content,&r.state.content_connected,&r.state.initial,&r.state.initial_connected,
      &r.messages,&r.connected,&r.scale_partials})receive(*t);
    response.emplace_back(local_error,owner.error);owner.response=std::make_unique<PeerExchange>(std::move(response),s.budget/4);
    owner.response->append_send(remote);remote.branch(again,{head});remote.mark(done);owner.reverse->append_finish(remote,local_error);
    owner.finish=std::make_unique<PeerExchange>(PeerExchange::Fields{{local_error,owner.error}},64);owner.finish->append_send(remote);remote.finish();
    p.copy(owner.command,work);owner.request->append_send(p);
  }
  for(auto& owner:s.shards) {
    if(owner.response)owner.response->append_receive(p);
    append_state_reverse_merge(p,owner.stage,owner.result,owner.ids,out,owner.program?owner.error:s.error,s.error);
  }
  return out;
}
void ShardedStateVjp::append_sources(DeviceProgram& p,const at::Tensor& messages,const at::Tensor& on,const at::Tensor& partials) {
  for(auto& owner:impl_->shards)append_state_reverse_sources(p,owner.packed,owner.result,messages,on,partials,impl_->error);
}
void ShardedStateVjp::append_stop(DeviceProgram& p) {
  auto& s=*impl_;
  for(auto& owner:s.shards)if(owner.request){p.copy(owner.command,owner.stop);owner.request->append_send(p);}
  for(auto& owner:s.shards)if(owner.finish){owner.finish->append_receive(p);append_state_reverse_error(p,owner.error,s.error);}
    else owner.reverse->append_finish(p,s.error);
}
void ShardedStateVjp::synchronize_inputs() const {
  c10::impl::VirtualGuardImpl(impl_->coordinator.type()).synchronizeDevice(impl_->coordinator.index());
  for(const auto& s:impl_->shards)if(s.program)c10::impl::VirtualGuardImpl(s.tape.state.decay.device().type()).synchronizeDevice(s.tape.state.decay.device().index());
}
void ShardedStateVjp::submit(){for(auto& s:impl_->shards)if(s.program)s.program->submit();}
void ShardedStateVjp::wait(){std::exception_ptr failure;for(auto& s:impl_->shards)if(s.program)try{s.program->wait();}catch(...){if(!failure)failure=std::current_exception();}if(failure)std::rethrow_exception(failure);}
void ShardedStateVjp::close(){for(auto& s:impl_->shards){if(s.program)s.program->close();for(auto* channel:{&s.init,&s.request,&s.response,&s.finish})if(*channel)(*channel)->close();}}
std::vector<StateShardGradient> ShardedStateVjp::gradients() const {std::vector<StateShardGradient> out;for(const auto& s:impl_->shards){if(!s.reverse)throw std::logic_error("state reverse not prepared");out.push_back(s.reverse->gradient());}return out;}
int64_t ShardedStateVjp::packet_bytes() const {int64_t n=0;for(const auto& s:impl_->shards)for(const auto* channel:{&s.init,&s.request,&s.response,&s.finish})if(*channel)n+=(*channel)->packet_bytes();return n;}
} // namespace tide::device_online
