#include "sharded_full_vjp.h"
#include "full_reverse_pack.h"
#include "peer_exchange.h"
#include <c10/core/impl/VirtualGuardImpl.h>
#include <ATen/core/grad_mode.h>
#include <set>
#include <stdexcept>

namespace tide::device_online {
namespace {
std::vector<at::Tensor*> parameters(FullVjp& g) {
  return {&g.weights,&g.biases,&g.extra.lh_weights,&g.extra.lh_biases,&g.extra.gate,&g.extra.up,&g.extra.down};
}
FullVjp accumulator(CannProgram& p,const FullTape& t) {
  FullVjp g{};auto floats=t.kinds.options().dtype(at::kFloat);
  auto make=[&](const at::Tensor& bank) {
    if(!bank.defined())return at::Tensor();auto shape=bank.sizes().vec();--shape[0];
    auto x=at::empty(shape,floats);p.zero(x);return x;
  };
  g.weights=make(t.weights);g.biases=make(t.biases);
  g.extra={make(t.extra.lh_weights),make(t.extra.lh_biases),make(t.extra.gate),make(t.extra.up),make(t.extra.down)};
  g.parameter_connected=at::empty({t.kinds.numel()},floats.dtype(at::kBool));g.chunks=at::empty({1},t.kinds.options());
  p.zero(g.parameter_connected);p.zero(g.chunks);return g;
}
void accumulate(CannProgram& p,FullVjp& total,FullVjp& partial,const at::Tensor& error) {
  auto a=parameters(total),b=parameters(partial);
  for(size_t i=0;i<a.size();++i)if(a[i]->defined())p.add(*a[i],*b[i]);
  append_connection_union(p,partial.parameter_connected,total.parameter_connected,error);p.add(total.chunks,partial.chunks);
}
void merge_error(CannProgram& p,const at::Tensor& src,const at::Tensor& dst) {
  auto zero=at::zeros_like(src),equal=at::empty({1},src.options().dtype(at::kBool)),branch=at::empty_like(src);
  p.equal(src,zero,equal);p.cast_index(equal,branch);auto failed=p.label(),done=p.label();
  p.branch(branch,{failed,done});p.mark(failed);p.copy(dst,src);p.mark(done);
}
}
struct ShardedFullVjp::Impl {
  struct Shard {
    FullShardTape tape;
    at::Tensor mapping,ids,work,command,stop;
    FullVjp total;
    std::unique_ptr<CannProgram> program;
    std::unique_ptr<PeerExchange> request,response;
  };
  at::Device coordinator;
  int64_t capacity,nodes,width,chunk,budget,workspace;
  bool constructed=false;
  std::vector<Shard> shards;
  Impl(at::Device d):coordinator(d){}
};
ShardedFullVjp::ShardedFullVjp(CannProgram& p,const ShardedReverseTape& t,int64_t chunk,int64_t budget,int64_t workspace)
  :impl_(std::make_unique<Impl>(t.coordinator.state.metadata.device())) {
  if(at::GradMode::is_enabled()||!t.coordinator.graph||t.coordinator.state.metadata.dim()!=2)
    throw std::invalid_argument("sharded reverse requires actual no-grad graph records");
  auto& s=*impl_;s.capacity=t.coordinator.state.metadata.size(0);s.nodes=t.coordinator.graph->nodes.size();s.width=t.coordinator.full.width;
  if(t.shards.empty()||budget<1||workspace<1||chunk<1)throw std::invalid_argument("invalid sharded Full reverse budgets");
  s.chunk=chunk;s.budget=budget/int64_t(t.shards.size());s.workspace=workspace;
  std::set<int64_t> nodes;std::set<c10::DeviceIndex> devices;
  for(const auto& shard:t.shards) {
    const auto& f=shard.full;
    if(!f.kinds.defined()||f.kinds.device().type()!=c10::DeviceType::PrivateUse1||f.kinds.dim()!=1||f.kinds.scalar_type()!=at::kLong
        ||!f.kinds.is_contiguous()||shard.nodes.empty()
        ||int64_t(shard.nodes.size())!=f.kinds.numel()||f.width!=s.width||f.samples!=t.coordinator.state.samples
        ||!devices.insert(f.kinds.device().index()).second)throw std::invalid_argument("invalid Full reverse shard layout");
    long double own=192.L*s.capacity*s.width+512.L*s.capacity+32.L*s.nodes+4096;
    for(const auto& x:{f.weights,f.biases,f.extra.lh_weights,f.extra.lh_biases,f.extra.gate,f.extra.up,f.extra.down})
      if(x.defined()) {
        if(x.device()!=f.kinds.device()||x.dim()<1||x.size(0)<1||!x.is_contiguous()||x.requires_grad()
            ||(x.scalar_type()!=at::kFloat&&x.scalar_type()!=at::kHalf))throw std::invalid_argument("invalid Full reverse parameter bank");
        own+=4.L*x.numel();
      }
    if(own>s.budget/2.L)throw std::invalid_argument("Full reverse packets/accumulators exceed budget");
    for(auto n:shard.nodes)if(n<0||n>=s.nodes||!nodes.insert(n).second)throw std::invalid_argument("duplicate or invalid Full reverse node");
  }
  if(int64_t(nodes.size())!=s.nodes)throw std::invalid_argument("incomplete Full reverse ownership");
  for(const auto& shard:t.shards) {
    Impl::Shard owner;owner.tape=shard;const auto device=shard.full.kinds.device();
    if(device!=s.coordinator){owner.program=std::make_unique<CannProgram>(device);owner.program->limit_workspace(workspace);}
    owner.total=accumulator(owner.program?*owner.program:p,shard.full);
    std::vector<int64_t> mapping(s.nodes,-1);for(size_t i=0;i<shard.nodes.size();++i)mapping[shard.nodes[i]]=i;
    owner.mapping=at::tensor(mapping,at::kLong).to(s.coordinator);owner.ids=at::tensor(shard.nodes,at::kLong).to(s.coordinator);
    owner.work=at::empty({2},owner.mapping.options());p.zero(owner.work);s.shards.push_back(std::move(owner));
  }
}
ShardedFullVjp::~ShardedFullVjp()=default;
FullVjp ShardedFullVjp::append_stage(CannProgram& p,const FullTape& stage,const at::Tensor& gradient,
    const at::Tensor& connected,const at::Tensor& error) {
  auto& s=*impl_;if(s.constructed)throw std::logic_error("sharded Full reverse stage already built");s.constructed=true;
  const auto capacity=s.capacity;auto floats=gradient.options();
  auto content=at::empty({2*capacity,s.width},floats),comparison=at::empty_like(content);
  auto con=at::empty({2*capacity},connected.options()),comp=at::empty_like(con);
  FullVjp out{content.narrow(0,0,capacity),comparison.narrow(0,0,capacity),con.narrow(0,0,capacity),comp.narrow(0,0,capacity),{},{},
    at::empty({s.nodes},connected.options()),at::empty({1},stage.count.options()),0};
  for(const auto& x:{content,comparison,con,comp,out.parameter_connected,out.chunks})p.zero(x);
  struct Stage {FullReverseBatch packed;at::Tensor error;FullVjp result;};std::vector<Stage> stages;
  for(auto& owner:s.shards) {
    auto batch=append_full_reverse_pack(p,stage,gradient,connected,owner.mapping,owner.tape.nodes.size(),owner.work,error);
    auto local_error=at::zeros_like(error);p.copy(local_error,error);
    auto tape=owner.tape.full;tape.metadata=batch.tape.metadata;tape.values=batch.tape.values;tape.count=batch.tape.count;batch.tape=tape;
    stages.push_back({batch,local_error,{}});
  }
  // Requests are issued for all nonempty peers before local work and waits.
  for(size_t i=0;i<s.shards.size();++i) {
    auto& owner=s.shards[i];auto& x=stages[i];if(!owner.program)continue;
    auto& remote=*owner.program;const auto device=owner.tape.full.kinds.device();
    auto buffer=[&](const at::Tensor& tensor){return at::zeros(tensor.sizes(),tensor.options().device(device));};
    owner.command=at::zeros_like(error);owner.stop=at::zeros_like(error);auto command=buffer(error),work=at::ones_like(error);
    auto tape=x.packed.tape;tape.metadata=buffer(tape.metadata);tape.values=buffer(tape.values);tape.count=buffer(tape.count);
    auto dy=buffer(gradient),on=buffer(connected),status=buffer(error);
    owner.request=std::make_unique<PeerExchange>(PeerExchange::Fields{{owner.command,command},{x.packed.tape.metadata,tape.metadata},
      {x.packed.tape.values,tape.values},{x.packed.tape.count,tape.count},{x.packed.gradient,dy},{x.packed.connected,on},{x.error,status}},s.budget/2);
    auto head=remote.label(),body=remote.label(),done=remote.label();auto again=at::zeros_like(command);
    remote.mark(head);owner.request->append_receive(remote);remote.branch(command,{done,body});remote.mark(body);
    auto local=append_full_vjp(remote,tape,dy,on,status,s.chunk,s.budget/2);accumulate(remote,owner.total,local,status);
    x.result.content=at::zeros_like(gradient);x.result.comparison=at::zeros_like(gradient);
    x.result.content_connected=at::zeros_like(connected);x.result.comparison_connected=at::zeros_like(connected);
    x.result.parameter_connected=at::zeros({int64_t(owner.tape.nodes.size())},connected.options());x.result.chunks=at::zeros_like(stage.count);
    owner.response=std::make_unique<PeerExchange>(PeerExchange::Fields{{local.content,x.result.content},{local.comparison,x.result.comparison},
      {local.content_connected,x.result.content_connected},{local.comparison_connected,x.result.comparison_connected},
      {local.parameter_connected,x.result.parameter_connected},{local.chunks,x.result.chunks},{status,x.error}},s.budget/2);
    owner.response->append_send(remote);remote.branch(again,{head});remote.mark(done);remote.finish();
    auto send=p.label(),skip=p.label();p.branch(x.packed.branch,{skip,send});p.mark(send);
    p.copy(owner.command,work);owner.request->append_send(p);p.mark(skip);
  }
  for(size_t i=0;i<s.shards.size();++i) {
    auto& owner=s.shards[i];auto& x=stages[i];if(owner.program)continue;
    x.result=append_full_vjp(p,x.packed.tape,x.packed.gradient,x.packed.connected,x.error,s.chunk,s.budget/2);
    accumulate(p,owner.total,x.result,x.error);
  }
  for(size_t i=0;i<s.shards.size();++i) {
    auto& owner=s.shards[i];auto& x=stages[i];auto receive=p.label(),skip=p.label();p.branch(x.packed.branch,{skip,receive});p.mark(receive);
    if(owner.response)owner.response->append_receive(p);merge_error(p,x.error,error);
    for(const auto& pair:{std::make_pair(content,x.result.content),std::make_pair(comparison,x.result.comparison),
        std::make_pair(con,x.result.content_connected),std::make_pair(comp,x.result.comparison_connected)})p.index_copy(pair.first,0,x.packed.destinations,pair.second);
    p.index_copy(out.parameter_connected,0,owner.ids,x.result.parameter_connected);p.add(out.chunks,x.result.chunks);p.mark(skip);
  }
  return out;
}
void ShardedFullVjp::append_stop(CannProgram& p) {for(auto& s:impl_->shards)if(s.request){p.copy(s.command,s.stop);s.request->append_send(p);}}
void ShardedFullVjp::synchronize_inputs() const {
  c10::impl::VirtualGuardImpl(impl_->coordinator.type()).synchronizeDevice(impl_->coordinator.index());
  for(const auto& s:impl_->shards)if(s.program)c10::impl::VirtualGuardImpl(s.tape.full.kinds.device().type()).synchronizeDevice(s.tape.full.kinds.device().index());
}
void ShardedFullVjp::submit(){for(auto& s:impl_->shards)if(s.program)s.program->submit();}
void ShardedFullVjp::wait(){std::exception_ptr failure;for(auto& s:impl_->shards)if(s.program)try{s.program->wait();}catch(...){if(!failure)failure=std::current_exception();}if(failure)std::rethrow_exception(failure);}
void ShardedFullVjp::close(){for(auto& s:impl_->shards){if(s.program)s.program->close();if(s.request)s.request->close();if(s.response)s.response->close();}}
std::vector<FullShardGradient> ShardedFullVjp::gradients() const {std::vector<FullShardGradient> out;for(const auto& s:impl_->shards)out.push_back({s.tape.nodes,s.total});return out;}
std::vector<at::Tensor> ShardedFullVjp::work() const {std::vector<at::Tensor> out;for(const auto& s:impl_->shards)out.push_back(s.work);return out;}
int64_t ShardedFullVjp::packet_bytes() const {int64_t n=0;for(const auto& s:impl_->shards)if(s.request)n+=s.request->packet_bytes()+s.response->packet_bytes();return n;}
ShardedGraphVjp append_sharded_graph_vjp(CannProgram& p,const ShardedReverseTape& t,const GraphCotangents& roots,
    const at::Tensor& error,int64_t chunk,int64_t budget,int64_t workspace) {
  auto full=std::make_shared<ShardedFullVjp>(p,t,chunk,budget/2,workspace);
  auto g=append_graph_vjp(p,t.coordinator,roots,error,chunk,budget/2,
    [full](CannProgram& p,const FullTape& stage,const at::Tensor& dy,const at::Tensor& on,const at::Tensor& error){return full->append_stage(p,stage,dy,on,error);});
  full->append_stop(p);return {std::move(g),std::move(full)};
}
void run_sharded_graph_vjp(CannProgram& p,const std::vector<ShardedGraphVjp>& gradients) {
  if(gradients.empty())throw std::invalid_argument("no sharded reverse programs");
  for(const auto& g:gradients)g.full->synchronize_inputs();p.submit();for(const auto& g:gradients)g.full->submit();
  std::exception_ptr failure;try{p.wait();}catch(...){failure=std::current_exception();}
  for(const auto& g:gradients)try{g.full->wait();}catch(...){if(!failure)failure=std::current_exception();}
  if(failure)std::rethrow_exception(failure);
}
} // namespace tide::device_online
