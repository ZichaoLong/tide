#include "weighted_flow_internal.h"
#include "tide/kernel.h"
#include "tide/next.h"
#include <ATen/core/grad_mode.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <limits>
#include <set>
#include <algorithm>
namespace tide::device_online {
namespace {
Model snapshot(Model m){
  std::map<const void*,at::Tensor> saved;
  auto copy=[&](at::Tensor& x){const auto id=x.unsafeGetTensorImpl();if(!saved.count(id))saved[id]=x.detach().clone().set_requires_grad(x.requires_grad());x=saved.at(id);};
  for(auto& w:m.nodes){for(auto x:{&w.decay,&w.weight,&w.bias,&w.read})copy(*x);for(auto& [_,x]:w.extra)copy(x);}
  for(auto& w:m.regions)for(auto& [_,x]:w.extra)copy(x);
  for(auto group:{&m.input_scale,&m.output_scale,&m.edge_scale,&m.agg_scale})for(auto& x:*group)copy(x);return m;
}
void fill(const AtomBatch& target,const std::vector<Atom>& atoms){
  if(atoms.size()>size_t(target.valid.numel()))throw std::invalid_argument("weighted boundary queue capacity exceeded");
  target.valid.zero_();if(atoms.empty())return;std::vector<int64_t> coords;std::vector<at::Tensor> values;
  for(const auto& a:atoms){coords.insert(coords.end(),{a.batch,a.node,a.time,a.kind,a.source,a.position});values.push_back(a.value);}
  target.coordinates.narrow(0,0,atoms.size()).copy_(at::tensor(coords,at::kLong).reshape({-1,6}));
  target.values.narrow(0,0,atoms.size()).copy_(at::stack(values));target.valid.narrow(0,0,atoms.size()).fill_(true);
}
void sync(at::Device d){c10::impl::VirtualGuardImpl(d.type()).synchronizeDevice(d.index());}
}
WeightedFlow::Impl::Impl(Graph g,Model m,const Continuation& q,at::Device d,WeightedFlowLimits l,std::vector<at::Device> placement)
  :graph(std::move(g)),model(std::move(m)),boundary(q),device(d),limits(l),node_devices(std::move(placement)){
  if(at::GradMode::is_enabled()||d.type()!=resident_device_type)throw std::invalid_argument("weighted flow requires explicit no-grad resident device");
  validate_kernel_device(d);
  if(l.queue<1||l.stages<1||l.cache<1||l.tensor_bytes<1||l.operator_bytes<1||l.timeout_ms<1||q.batch_size<1
    ||static_cast<long double>(l.queue)*q.batch_size*std::max<int64_t>(1,model.width())*64>l.tensor_bytes)
    throw std::invalid_argument("invalid weighted flow capacities/budget");
  graph.compile();configure_model(graph,model);validate_model(graph,model);
  if(node_devices.empty())node_devices.assign(graph.nodes.size(),d);
  if(node_devices.size()!=graph.nodes.size())throw std::invalid_argument("weighted placement needs one device per node");
  std::set<int64_t> checked;
  for(auto target:node_devices){
    if(target.type()!=d.type()||target.index()<0)throw std::invalid_argument("weighted node placement requires indexed same-backend devices");
    if(checked.insert(target.index()).second)validate_kernel_device(target);
  }
  if(l.timeout_ms>60000&&std::any_of(node_devices.begin(),node_devices.end(),[&](at::Device x){return x!=d;}))
    throw std::invalid_argument("weighted peer notification timeout must not exceed 60000ms");
  for(const auto& w:model.nodes)if(!w.bias.device().is_cpu()||(w.bias.scalar_type()!=at::kFloat&&w.bias.scalar_type()!=at::kHalf)||!w.next_kernel->comparison_identity())
    throw std::invalid_argument("weighted flow requires CPU FP32/FP16 parameters and comparison-identity Next");
  journal_bytes=weighted_flow_preflight(graph,model,q.batch_size,l);
  model=snapshot(std::move(model));validate_window(graph,model,boundary,{},q.cut,q.cut);
  for(size_t v=0;v<graph.nodes.size();++v){auto n=std::make_unique<WeightedFlowNode>(model,v,boundary,node_devices[v],l.cache,l.tensor_bytes);
    n->binding.sources.validate(graph);const auto r=graph.nodes[v].region;
    if(!weighted::same_configuration(n->binding.regional,weighted::region_binding(model.regions.at(r))))
      throw std::invalid_argument("weighted node Describe and region configuration differ");
    nodes.push_back(std::move(n));
  }
  for(size_t r=0;r<graph.regions.size();++r)regions.push_back(std::make_unique<WeightedFlowRegion>(graph,model,r,boundary,d));
  const auto opts=at::TensorOptions().dtype(model.nodes.at(0).bias.scalar_type()).device(d);
  error=at::zeros({1},opts.dtype(at::kInt));stop=at::full({1},q.cut,opts.dtype(at::kLong));stages=at::zeros_like(stop);begin=stop.clone();
  pending=std::make_unique<QueueTransaction>(l.queue,model.width(),graph.nodes.size(),q.batch_size,opts,error);
  external={at::zeros({l.queue,6},opts.dtype(at::kLong)),at::zeros({l.queue,model.width()},opts),at::zeros({l.queue},opts.dtype(at::kBool))};
  fill(pending->atoms(),q.pending);pending->stats()[0]=q.pending.size();pending->stats()[1]=q.pending.size();
  journal=std::make_unique<WeightedWindowJournal>(l.stages,journal_bytes);construct();
  boundary.states.clear();boundary.history.clear();boundary.pending.clear();
}
void WeightedFlow::Impl::construct(){
  std::vector<int64_t> owners;std::vector<Wire> wires;
  for(const auto& n:graph.nodes)owners.push_back(n.region);for(const auto& e:graph.edges)wires.push_back({e.source,e.target,e.delay});
  // Local memory components expose one time per private state per stage. The
  // closure still batches independent complete frames on the actual queue.
  DeviceReady ready(owners,graph.regions.size(),wires,boundary.batch_size,device,limits.prefill,std::vector<int64_t>(regions.size(),1));
  BroadcastRouter router(nodes.size(),boundary.batch_size,wires,limits.queue,device);
  program=std::make_unique<DeviceProgram>(device);auto& p=*program;p.limit_workspace(limits.operator_bytes);
  services=std::make_unique<TaskServices>(p,device,limits.operator_bytes);
  peers=std::make_unique<WeightedFlowPeers>(device,limits);
  for(size_t v=0;v<nodes.size();++v)if(node_devices[v]==device){auto& worker=services->begin();
    nodes[v]->source=bind_weighted_flow_source(worker,nodes[v]->banks,graph,v,nodes[v]->binding,error,remaining(worker));services->end(worker);}
  peers->bind(graph,nodes,[&](DeviceProgram& worker){return remaining(worker);});
  for(const auto& n:nodes)input_classes.push_back(n->source.classes.to(device));
  views.resize(nodes.size());peers->append_begin(p,error);

  journal->capture(p,"begin",begin);
  journal->capture(p,"pending.meta",pending->atoms().coordinates);journal->capture(p,"pending.present",pending->atoms().valid);
  for(size_t v=0;v<nodes.size();++v){const auto prefix="node."+std::to_string(v)+".";
    journal->capture(p,prefix+"integer",node_devices[v]==device?nodes[v]->state.integer:peers->boundary_integer(v));
    journal->capture(p,prefix+"present",node_devices[v]==device?nodes[v]->state_present:peers->boundary_present(v));}
  pending->append_stage(p,at::zeros({limits.queue},error.options()),external);
  auto cap=at::full_like(stages,limits.stages),one=at::ones_like(stages),index=at::zeros_like(error),budget_error=at::full_like(error,4);
  auto predicate=at::zeros({1},error.options().dtype(at::kBool));
  auto head=p.label(),test=p.label(),body=p.label(),exhausted=p.label(),end=p.label();
  p.mark(head);auto frame=ready.append_stage(p,pending->atoms(),stop,error);p.branch(frame.branch,{end,test});
  p.mark(test);p.less(stages,cap,predicate);p.cast_index(predicate,index);p.branch(index,{exhausted,body});p.mark(body);
  append_nodes(p,frame);append_regions(p);auto actions=append_full(p);
  auto arrivals=router.append_stage(p,actions,at::ones({std::max<int64_t>(1,graph.edges.size()),1},actions.values.options()),error);
  auto transaction=pending->propose_stage(p,frame.consumed,arrivals);
  save_stage(p,frame,arrivals);
  // Every numerical/capacity preflight completes before any persistent owner.
  pending->commit_stage(p,transaction);
  for(size_t v=0;v<nodes.size();++v)if(node_devices[v]==device){auto& n=*nodes[v];
    append_weighted_raw_adopt(p,n.state,n.proposal,error);append_weighted_owner_adopt(p,n.state_present,n.input.present,n.input.times,error,false);}
  peers->append_commit(p,nodes,error);
  for(auto& r:regions)append_weighted_owner_adopt(p,r->history,r->selection->history_present,r->selection->history_times,error,true);
  p.add(stages,one);p.branch(index,{head});p.mark(exhausted);weighted_copy(p,error,budget_error);p.mark(end);
  peers->append_stop(p);services->append_stop();bytes=limits.tensor_bytes-remaining(p);peers->finish();services->finish();p.finish();
  bytes=limits.tensor_bytes-remaining(p);
}
WeightedFlow::WeightedFlow(Graph g,Model m,const Continuation& q,at::Device d,WeightedFlowLimits l)
  :WeightedFlow(std::move(g),std::move(m),q,d,l,{}){}
WeightedFlow::WeightedFlow(Graph g,Model m,const Continuation& q,at::Device d,WeightedFlowLimits l,std::vector<at::Device> placement)
  :impl_(std::make_unique<Impl>(std::move(g),std::move(m),q,d,l,std::move(placement))){}
WeightedFlow::~WeightedFlow()=default;
Result WeightedFlow::advance(const std::vector<External>& xs,Index until){
  if(!impl_||impl_->failed)throw std::logic_error("weighted flow closed/failed; restore an earlier checkpoint into a new owner");
  impl_->execute(xs,until);return impl_->decode();
}
void WeightedFlow::Impl::execute(const std::vector<External>& xs,Index until){
  if(failed)throw std::logic_error("weighted flow failed; restore an earlier checkpoint into a new owner");
  if(at::GradMode::is_enabled())throw std::invalid_argument("weighted inference owner requires no-grad");auto& s=*this;
  if(xs.size()>size_t(s.limits.queue))throw std::invalid_argument("weighted external input capacity exceeded");
  auto input=validate_external(s.graph,s.model,s.boundary,xs,until,until); // Retryable boundary rejection.
  try {
    fill(s.external,input.atoms);s.begin.fill_(s.boundary.cut);s.stop.fill_(until);s.stages.zero_();++s.generation;
    s.peers->synchronize_inputs();sync(s.device);s.peers->submit();s.services->submit();s.program->run(s.limits.timeout_ms);
    s.services->wait(s.limits.timeout_ms);s.peers->wait();
    const auto code=s.error.item<int32_t>();if(code)throw std::runtime_error("weighted resident transaction refused, device error="+std::to_string(code));
    s.boundary.cut=until;for(const auto& [owner,last]:input.ledger_updates)s.boundary.ledger[owner]=last;s.ran=true;
  }catch(...){s.failed=true;throw;}
}
Continuation WeightedFlow::checkpoint()const{
  if(!impl_||at::GradMode::is_enabled())throw std::logic_error("weighted checkpoint requires live no-grad owner");return impl_->checkpoint();
}
int64_t WeightedFlow::reserved_bytes()const{if(!impl_)throw std::logic_error("closed weighted flow");return impl_->bytes;}
std::vector<at::Device> WeightedFlow::node_devices()const{if(!impl_)throw std::logic_error("closed weighted flow");return impl_->node_devices;}
void WeightedFlow::close(){if(impl_){impl_->program->close();impl_->services->close();impl_->peers->close();impl_.reset();}}
}
