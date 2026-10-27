#include "weighted_flow_peers.h"
#include "weighted_flow_internal.h"
#include "weighted_flow_vjp.h"
#include "peer_exchange.h"
#include <c10/core/impl/VirtualGuardImpl.h>
namespace tide::device_online {
namespace {
// Preserve the first failure in deterministic owner order; no host scalar.
void merge(DeviceProgram& p,const at::Tensor& from,const at::Tensor& error){
  auto ok=at::zeros({1},error.options().dtype(at::kBool)),index=at::zeros_like(error);
  auto copy=p.label(),done=p.label();p.equal(error,at::zeros_like(error),ok);p.cast_index(ok,index);
  p.branch(index,{done,copy});p.mark(copy);p.copy(error,from);p.mark(done);
}
using Exchange=std::unique_ptr<PeerExchange>;
struct Shard {
  at::Device device;
  std::vector<int64_t> nodes;
  at::Tensor error,status,command,remote_command,stop,work;
  std::unique_ptr<DeviceProgram> program;
  std::unique_ptr<TaskServices> services;
  Exchange initial,request,read,selection,full,decision,completion,roots,gradient;
  size_t head=0,end=0;
  explicit Shard(at::Device d):device(d){}
};
}
struct WeightedFlowPeers::Impl {
  at::Device coordinator;WeightedFlowLimits limits;
  std::vector<std::unique_ptr<Shard>> shards;
  std::vector<at::Tensor> held;
  std::map<int64_t,at::Tensor> integer,present;
  Budget remaining;
  Impl(at::Device d,WeightedFlowLimits l):coordinator(d),limits(l){}
  at::Tensor mirror(const at::Tensor& x,at::Device d,DeviceProgram& p){
    if(x.nbytes()>static_cast<uint64_t>(remaining(p)))throw std::invalid_argument("weighted peer mirror exceeds tensor budget before allocation");
    auto result=at::zeros(x.sizes(),x.options().device(d));held.push_back(result);return result;
  }
  Exchange exchange(PeerExchange::Fields fields,DeviceProgram& p){
    return std::make_unique<PeerExchange>(std::move(fields),remaining(p),limits.timeout_ms);
  }
};
WeightedFlowPeers::WeightedFlowPeers(at::Device d,const WeightedFlowLimits& l):impl_(std::make_unique<Impl>(d,l)){}
WeightedFlowPeers::~WeightedFlowPeers()=default;
void WeightedFlowPeers::bind(const Graph& g,Nodes& nodes,const Budget& remaining){
  auto& s=*impl_;s.remaining=remaining;
  for(size_t v=0;v<nodes.size();++v){const auto d=nodes[v]->state.floating.device();if(d==s.coordinator)continue;
    Shard* shard=nullptr;for(auto& candidate:s.shards)if(candidate->device==d)shard=candidate.get();
    if(!shard){auto owned=std::make_unique<Shard>(d);shard=owned.get();s.shards.push_back(std::move(owned));
      auto opts=at::TensorOptions().device(d).dtype(at::kInt);shard->error=at::zeros({1},opts);
      shard->status=at::zeros({1},opts.device(s.coordinator));
      shard->program=std::make_unique<DeviceProgram>(d);shard->program->limit_workspace(s.limits.operator_bytes);
      shard->services=std::make_unique<TaskServices>(*shard->program,d,s.limits.operator_bytes);
    }
    shard->nodes.push_back(v);auto& worker=shard->services->begin();auto& n=*nodes[v];
    n.source=bind_weighted_flow_source(worker,n.banks,g,v,n.binding,shard->error,remaining(worker));shard->services->end(worker);
  }
  for(auto& shard:s.shards){auto& a=*shard;PeerExchange::Fields fields{{a.error,a.status}};
    for(auto v:a.nodes){const auto& n=*nodes[v];
      s.integer[v]=s.mirror(n.state.integer,s.coordinator,*a.program);
      s.present[v]=s.mirror(n.state_present,s.coordinator,*a.program);
      fields.push_back({n.state.integer,s.integer[v]});fields.push_back({n.state_present,s.present[v]});
    }
    a.initial=s.exchange(std::move(fields),*a.program);a.initial->append_send(*a.program);
  }
}
void WeightedFlowPeers::append_begin(DeviceProgram& p,const at::Tensor& error){
  for(auto& a:impl_->shards){a->initial->append_receive(p);merge(p,a->status,error);}
}
void WeightedFlowPeers::append_local_send(DeviceProgram& coordinator,const Graph& g,Nodes& nodes,
    const std::vector<WeightedNodeInput>& inputs,std::vector<WeightedFlowView>& views,const at::Tensor& error,const Budget& remaining,
    const std::vector<WeightedRawState>* saved_old){
  auto& s=*impl_;
  for(auto& owned:s.shards){auto& a=*owned;auto& p=*a.program;
    a.command=at::zeros_like(error);a.stop=at::zeros_like(error);a.work=at::ones_like(error);
    a.remote_command=s.mirror(a.command,a.device,p);
    PeerExchange::Fields request{{a.command,a.remote_command},{error,a.error}};
    for(auto v:a.nodes){auto& n=*nodes[v];const auto& input=inputs[v];
      auto copy=[&](const at::Tensor& x){auto y=s.mirror(x,a.device,p);request.push_back({x,y});return y;};
      n.input={copy(input.payload),copy(input.owners),copy(input.lengths),copy(input.times),copy(input.present),copy(input.bank),copy(input.indices)};
      if(saved_old){const auto& old=saved_old->at(v);
        request.push_back({old.floating,n.state.floating});request.push_back({old.integer,n.state.integer});}
    }
    a.request=s.exchange(std::move(request),p);a.head=p.label();const auto body=p.label();a.end=p.label();
    p.mark(a.head);a.request->append_receive(p);p.branch(a.remote_command,{a.end,body});p.mark(body);
    PeerExchange::Fields result{{a.error,a.status}};
    for(auto v:a.nodes){auto& n=*nodes[v];append_weighted_node_local(p,*a.services,n,g.regions[g.nodes[v].region].read_mode,a.error,remaining);
      auto& view=views[v];view.present=inputs[v].present;view.times=inputs[v].times;view.bank=inputs[v].bank;
      auto back=[&](const at::Tensor& x){auto y=s.mirror(x,s.coordinator,p);result.push_back({x,y});return y;};
      view.content=back(n.aggregate->output);view.old={back(n.old.floating),back(n.old.integer)};
      view.proposal={back(n.proposal.floating),back(n.proposal.integer)};
      for(const auto& [name,value]:n.describe->fields)view.describe[name]=back(value);
    }
    a.read=s.exchange(std::move(result),p);a.read->append_send(p);
    coordinator.copy(a.command,a.work);a.request->append_send(coordinator);
  }
}
void WeightedFlowPeers::append_local_receive(DeviceProgram& p,const at::Tensor& error){
  for(auto& a:impl_->shards){a->read->append_receive(p);merge(p,a->status,error);}
}
void WeightedFlowPeers::append_full(DeviceProgram& coordinator,Nodes& nodes,std::vector<WeightedFlowView>& views,
    const at::Tensor& error,const Budget& remaining){
  auto& s=*impl_;
  for(auto& owned:s.shards){auto& a=*owned;auto& p=*a.program;
    PeerExchange::Fields selected{{error,a.error}},result{{a.error,a.status}};
    for(auto v:a.nodes){auto& n=*nodes[v];auto& view=views[v];
      n.control=s.mirror(view.control,a.device,p);n.active=s.mirror(view.active,a.device,p);
      selected.push_back({view.control,n.control});selected.push_back({view.active,n.active});
    }
    a.selection=s.exchange(std::move(selected),p);a.selection->append_receive(p);
    for(auto v:a.nodes){auto& n=*nodes[v];auto& view=views[v];append_weighted_node_full(*a.services,n,a.error,remaining);
      n.payload=append_weighted_payload(p,n.control,n.full,n.active,n.state.floating.scalar_type(),a.error);
      view.full=s.mirror(n.full,s.coordinator,p);view.payload=s.mirror(n.payload,s.coordinator,p);
      result.push_back({n.full,view.full});result.push_back({n.payload,view.payload});
    }
    a.full=s.exchange(std::move(result),p);a.full->append_send(p);a.selection->append_send(coordinator);
  }
  for(auto& a:s.shards){a->full->append_receive(coordinator);merge(coordinator,a->status,error);}
}
void WeightedFlowPeers::append_commit(DeviceProgram& coordinator,Nodes& nodes,const at::Tensor& error){
  auto& s=*impl_;
  for(auto& owned:s.shards){auto& a=*owned;auto& p=*a.program;
    a.decision=s.exchange({{error,a.error}},p);a.decision->append_receive(p);
    for(auto v:a.nodes){auto& n=*nodes[v];append_weighted_raw_adopt(p,n.state,n.proposal,a.error);
      append_weighted_owner_adopt(p,n.state_present,n.input.present,n.input.times,a.error,false);}
    a.completion=s.exchange({{a.error,a.status}},p);a.completion->append_send(p);
    p.branch(at::zeros_like(a.error),{a.head});p.mark(a.end);
    a.decision->append_send(coordinator);a.completion->append_receive(coordinator);
  }
}
void WeightedFlowPeers::append_reverse(DeviceProgram& coordinator,const Graph& graph,Nodes& nodes,
    const std::vector<WeightedNodeRoots>& roots,std::vector<WeightedNodeGradient>& gradients,
    const at::Tensor& error,const Budget& remaining){
  auto& s=*impl_;
  for(auto& owned:s.shards){auto& a=*owned;auto& p=*a.program;
    PeerExchange::Fields request{{error,a.error}},result{{a.error,a.status}};
    std::vector<WeightedNodeRoots> local;
    auto copy=[&](const at::Tensor& x){auto y=s.mirror(x,a.device,p);request.push_back({x,y});return y;};
    auto row=[&](const WeightedRowAdjoint& x){return WeightedRowAdjoint{copy(x.value),copy(x.connected)};};
    for(auto v:a.nodes){const auto& r=roots.at(v);WeightedNodeRoots owned;
      owned.content=row(r.content);owned.full=row(r.full);owned.proposal={copy(r.proposal.value),copy(r.proposal.connected)};
      for(const auto& [name,x]:r.describe)owned.describe[name]=row(x);local.push_back(std::move(owned));}
    a.roots=s.exchange(std::move(request),p);a.roots->append_receive(p);
    auto back=[&](const at::Tensor& x){auto y=s.mirror(x,s.coordinator,p);result.push_back({x,y});return y;};
    for(size_t j=0;j<a.nodes.size();++j){const auto v=a.nodes[j];auto& n=*nodes[v];
      // Match the coordinator's per-node service boundary. Appending every
      // node's raw/parameter reductions to this peer's control stream can
      // exhaust CANN's task buffer even when each numerical component fits.
      auto& worker=a.services->begin();
      auto g=append_weighted_node_vjp(worker,*a.services,n,graph.regions[graph.nodes[v].region].read_mode,local[j],a.error,remaining);
      a.services->end(worker);
      auto& returned=gradients.at(v);returned.old={back(g.old.value),back(g.old.connected)};
      returned.initial={back(g.initial.value),back(g.initial.connected)};
      returned.payload=back(g.payload);returned.connected=back(g.connected);
      for(const auto& part:g.parameters)returned.parameters.push_back({part.name,back(part.values),back(part.connected)});
    }
    a.gradient=s.exchange(std::move(result),p);a.gradient->append_send(p);
    p.branch(at::zeros_like(a.error),{a.head});p.mark(a.end);a.roots->append_send(coordinator);
  }
  for(auto& a:s.shards){a->gradient->append_receive(coordinator);merge(coordinator,a->status,error);}
}
void WeightedFlowPeers::append_stop(DeviceProgram& p){for(auto& a:impl_->shards){p.copy(a->command,a->stop);a->request->append_send(p);}}
void WeightedFlowPeers::finish(){for(auto& a:impl_->shards){a->services->append_stop();a->services->finish();a->program->finish();}}
void WeightedFlowPeers::synchronize_inputs(){for(auto& a:impl_->shards){a->error.zero_();c10::impl::VirtualGuardImpl(a->device.type()).synchronizeDevice(a->device.index());}}
void WeightedFlowPeers::submit(){for(auto& a:impl_->shards){a->services->submit();a->program->submit();}}
void WeightedFlowPeers::wait(){std::exception_ptr error;
  for(auto& a:impl_->shards)try{a->program->wait(impl_->limits.timeout_ms);a->services->wait(impl_->limits.timeout_ms);}catch(...){if(!error)error=std::current_exception();}
  if(error)std::rethrow_exception(error);
}
void WeightedFlowPeers::close(){for(auto& a:impl_->shards){a->program->close();a->services->close();
  for(auto* x:{a->initial.get(),a->request.get(),a->read.get(),a->selection.get(),a->full.get(),a->decision.get(),a->completion.get(),a->roots.get(),a->gradient.get()})if(x)x->close();}}
std::vector<const DeviceProgram*> WeightedFlowPeers::programs()const{
  std::vector<const DeviceProgram*> result;for(const auto& a:impl_->shards)if(a->services){auto ps=a->services->programs();result.insert(result.end(),ps.begin(),ps.end());result.push_back(a->program.get());}return result;
}
std::vector<at::Tensor> WeightedFlowPeers::tensors()const{
  auto out=impl_->held;for(const auto& a:impl_->shards)for(const auto& t:{a->error,a->status,a->command,a->remote_command,a->stop,a->work})if(t.defined())out.push_back(t);return out;
}
at::Tensor WeightedFlowPeers::boundary_integer(int64_t node)const{return impl_->integer.at(node);}
at::Tensor WeightedFlowPeers::boundary_present(int64_t node)const{return impl_->present.at(node);}
int64_t WeightedFlowPeers::size()const{return impl_->shards.size();}
int64_t WeightedFlowPeers::workspace_bytes()const{int64_t out=0;for(const auto& a:impl_->shards)out+=a->services->workspace_bytes();return out;}
int64_t WeightedFlowPeers::packet_bytes()const{int64_t out=0;for(const auto& a:impl_->shards)
  for(auto* x:{a->initial.get(),a->request.get(),a->read.get(),a->selection.get(),a->full.get(),a->decision.get(),a->completion.get(),a->roots.get(),a->gradient.get()})if(x)out+=x->packet_bytes();return out;}
}
