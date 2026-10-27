// Finite-cut chain rule and actual-journal contract: docs/resident-weighted-window-vjp.md.
#include "weighted_window_vjp.h"
#include <ATen/core/grad_mode.h>
namespace tide::device_online {
WeightedWindowAdjoint::WeightedWindowAdjoint(WeightedFlow& forward):WeightedWindowAdjoint(forward,0){}
WeightedWindowAdjoint::WeightedWindowAdjoint(WeightedFlow& forward,int64_t budget):forward_(forward){
  if(at::GradMode::is_enabled()||!forward.impl_||forward.impl_->failed||!forward.impl_->ran)
    throw std::invalid_argument("weighted window adjoint requires a successful live no-grad forward window");
  auto& s=*forward.impl_;auto& journal=*s.journal;generation_=s.generation;
  if(budget<0)throw std::invalid_argument("negative weighted window total tensor budget");
  budget_=budget?budget:s.limits.tensor_bytes;
  const auto device=s.device;const auto options=s.error.options();const auto samples=s.boundary.batch_size;
  const auto count=int64_t(s.nodes.size()),capacity=s.limits.queue,stages=s.limits.stages,width=s.model.width();
  long double root_bytes=static_cast<long double>(capacity)*(1+4.L*width)*(1+2.L*stages);
  root_bytes+=static_cast<long double>(stages)*count*samples*(10.L+4.L*width);
  root_bytes+=2.L*capacity*(50.L+4.L*width)+16.L*samples*s.regions.size();
  for(size_t v=0;v<s.nodes.size();++v){const auto& n=*s.nodes[v];const auto prefix="node."+std::to_string(v)+".";
    root_bytes+=3.L*samples*(4.L*n.layout.floating_width()+8.L*n.layout.integer_width())+8.L*samples;
    for(const auto& [name,role]:n.banks.roles())root_bytes+=role.bank.nbytes();
    root_bytes+=(1+2.L*stages)*samples*(4.L*n.layout.floating_width()+n.layout.fields().size());
    for(const std::string name:{"content","control","full"})root_bytes+=4.L*journal.field(prefix+name).numel()+stages*samples;
    for(const auto& [name,x]:n.describe->fields)root_bytes+=4.L*journal.field(prefix+"describe."+name).numel()+stages*samples;
  }
  for(const auto& region:s.regions)for(const auto& [name,role]:region->banks.roles())root_bytes+=role.bank.nbytes();
  if(root_bytes+s.bytes>=budget_)throw std::invalid_argument("weighted window roots exceed total tensor budget before allocation");
  program_=std::make_unique<DeviceProgram>(device);auto& p=*program_;p.limit_workspace(s.limits.operator_bytes);
  services_=std::make_unique<TaskServices>(p,device,s.limits.operator_bytes);auto& services=*services_;
  error_=at::zeros_like(s.error);stage_=at::zeros_like(s.stages);const auto& error=error_;
  Continuation initial;initial.identity=s.graph.identity;initial.batch_size=samples;initial.cut=0;
  auto remaining=[&](DeviceProgram& worker){return this->remaining(worker);};
  for(int64_t v=0;v<count;++v){nodes_.push_back(std::make_unique<WeightedFlowNode>(s.model,v,initial,device,s.limits.cache,budget_));auto& n=*nodes_.back();
    auto& worker=services.begin();n.source=bind_weighted_flow_source(worker,n.banks,s.graph,v,n.binding,error,remaining(worker));services.end(worker);
    const auto prefix="node."+std::to_string(v)+".";WeightedWindowNodeRoots r;
    r.content=weighted_window::row(journal.field(prefix+"content"));r.control=weighted_window::row(journal.field(prefix+"control").unsqueeze(-1));
    r.full=weighted_window::row(journal.field(prefix+"full"));r.old=weighted_window::raw(n.layout,journal.field(prefix+"old.float"));
    r.proposal=weighted_window::raw(n.layout,journal.field(prefix+"proposal.float"));r.final=weighted_window::raw(n.layout,n.state.floating,1);
    for(const auto& [name,x]:s.nodes[v]->describe->fields)r.describe[name]=weighted_window::row(journal.field(prefix+"describe."+name));
    roots_.nodes.push_back(std::move(r));
  }
  for(size_t r=0;r<s.regions.size();++r)regions_.push_back(std::make_unique<WeightedFlowRegion>(s.graph,s.model,r,initial,device));
  roots_.pending=weighted_window::row(s.pending->atoms().values,1);
  roots_.messages=weighted_window::row(journal.field("message.values"));roots_.fibers=weighted_window::row(journal.field("fiber.values"));
  WeightedReverseEvents events{at::zeros({stages,count,samples},options.dtype(at::kLong)),at::zeros({stages,count,samples},options.dtype(at::kBool)),
    at::zeros({stages,count,samples,width},options.dtype(at::kFloat)),at::zeros({stages,count,samples},options.dtype(at::kBool))};
  for(int64_t v=0;v<count;++v){const auto prefix="node."+std::to_string(v)+".";
    weighted_copy(p,events.times.select(1,v),journal.field(prefix+"times"));weighted_copy(p,events.present.select(1,v),journal.field(prefix+"active"));}
  boundary_={at::zeros({2*capacity,6},options.dtype(at::kLong)),at::zeros({2*capacity},options.dtype(at::kBool)),
    at::zeros({2*capacity,width},options.dtype(at::kFloat)),at::zeros({2*capacity},options.dtype(at::kBool))};
  weighted_copy(p,boundary_.coordinates.narrow(0,0,capacity),journal.boundary("pending.meta"));
  weighted_copy(p,boundary_.present.narrow(0,0,capacity),journal.boundary("pending.present"));
  weighted_copy(p,boundary_.coordinates.narrow(0,capacity,capacity),s.external.coordinates);
  weighted_copy(p,boundary_.present.narrow(0,capacity,capacity),s.external.valid);
  std::vector<int64_t> edge_sources;for(const auto& e:s.graph.edges)edge_sources.push_back(e.source);
  auto edges=at::tensor(edge_sources,at::kLong).to(device);const auto& begin=journal.boundary("begin");
  auto zero=at::zeros_like(stage_),minus=at::full_like(stage_,-1),index=at::zeros_like(error),predicate=at::zeros({1},options.dtype(at::kBool));
  auto init=p.label(),test=p.label(),body=p.label(),done=p.label();p.branch(index,{init});p.mark(test);
  p.less(zero,stage_,predicate);p.cast_index(predicate,index);p.branch(index,{done,body});p.mark(body);p.add(stage_,minus);
  auto emitted=AtomBatch{journal.load(p,stage_,"message.meta"),journal.load(p,stage_,"message.values"),journal.load(p,stage_,"message.present")};
  auto after=at::zeros_like(stage_);p.copy(after,stage_);p.add(after,at::ones_like(after));
  weighted_window::route(p,emitted,weighted_window::load(p,stage_,roots_.messages),edges,events,boundary_,after,begin,error);
  ReadyBatch ready;ready.atoms={journal.load(p,stage_,"fiber.meta"),journal.load(p,stage_,"fiber.values"),journal.load(p,stage_,"fiber.present")};
  ready.fibers=journal.load(p,stage_,"fiber.rows");ready.fiber_offsets=journal.load(p,stage_,"fiber.offsets");ready.counts=journal.load(p,stage_,"counts");
  weighted_window::route(p,ready.atoms,weighted_window::load(p,stage_,roots_.fibers),edges,events,boundary_,stage_,begin,error);
  for(int64_t v=0;v<count;++v){auto& n=*nodes_[v];const auto prefix="node."+std::to_string(v)+".";
    auto& worker=services.begin();auto& p=worker;
    weighted_copy(p,n.state.floating,journal.load(p,stage_,prefix+"old.float"));weighted_copy(p,n.state.integer,journal.load(p,stage_,prefix+"old.int"));
    n.input=append_weighted_node_input(p,ready,n.source.classes,v,samples,s.graph.inputs.size(),error);
    append_weighted_node_local(p,services,n,s.graph.regions[s.graph.nodes[v].region].read_mode,error,remaining);
    services.end(worker);
  }
  for(auto& r:regions_){auto& worker=services.begin();append_weighted_region_local(worker,services,*r,nodes_,error,remaining);services.end(worker);}
  for(auto& n:nodes_)append_weighted_node_full(services,*n,error,remaining);
  auto event_values=weighted_window::load(p,stage_,events.values),event_flags=weighted_window::load(p,stage_,events.connected);
  std::map<int64_t,WeightedRowAdjoint> controls;std::vector<WeightedNodeRoots> locals;
  std::vector<WeightedRawAdjoint> rolling;std::vector<WeightedParameterPartial> partials;
  for(int64_t v=0;v<count;++v){auto& n=*nodes_[v];const auto& r=roots_.nodes[v];
    auto& worker=services.begin();auto& p=worker;
    rolling.push_back({at::zeros_like(n.state.floating),at::zeros({samples,int64_t(n.layout.fields().size())},options.dtype(at::kBool))});
    auto control=weighted_reverse::zeros(p,at::zeros({samples,1},options.dtype(at::kFloat)));
    weighted_reverse::add(p,control,weighted_window::load(p,stage_,r.control),n.input.bank,error);
    weighted_reverse::add(p,control,{weighted_columns(p,event_values[v],0,1),event_flags[v]},n.input.bank,error);controls[v]=control;
    WeightedNodeRoots local;local.content=weighted_window::load(p,stage_,r.content);local.full=weighted_reverse::zeros(p,n.full);
    weighted_reverse::add(p,local.full,weighted_window::load(p,stage_,r.full),n.input.bank,error);
    weighted_reverse::add(p,local.full,{weighted_columns(p,event_values[v],1,width-1),event_flags[v]},n.input.bank,error);
    local.proposal=weighted_reverse::zeros(p,n.layout,n.state.floating);
    weighted_reverse::add(p,n.layout,local.proposal,rolling.back(),error);
    weighted_reverse::add(p,n.layout,local.proposal,weighted_window::load(p,stage_,r.proposal),error);
    for(const auto& [name,g]:r.describe)local.describe[name]=weighted_window::load(p,stage_,g);
    locals.push_back(std::move(local));
    services.end(worker);
  }
  for(auto& r:regions_){auto& worker=services.begin();auto& p=worker;
    auto g=append_weighted_flow_region_vjp(p,services,*r,nodes_,controls,error,remaining);
    partials.insert(partials.end(),g.parameters.begin(),g.parameters.end());
    for(const auto& [v,fields]:g.describe)for(const auto& [name,part]:fields){auto combined=weighted_reverse::zeros(p,part.value);
      weighted_reverse::add(p,combined,locals[v].describe.at(name),nodes_[v]->input.bank,error);
      weighted_reverse::add(p,combined,part,nodes_[v]->input.bank,error);locals[v].describe[name]=combined;}services.end(worker);}
  for(int64_t v=0;v<count;++v){auto& n=*nodes_[v];const auto& r=roots_.nodes[v];
    auto& worker=services.begin();auto& p=worker;
    auto g=append_weighted_node_vjp(p,services,n,s.graph.regions[s.graph.nodes[v].region].read_mode,locals[v],error,remaining);
    weighted_reverse::add(p,n.layout,g.old,weighted_window::load(p,stage_,r.old),error);
    weighted_copy(p,rolling[v].value,g.old.value);weighted_copy(p,rolling[v].connected,g.old.connected);
    append_weighted_reverse_route(p,ready.atoms,n.input.indices,n.input.lengths,g.payload,g.connected,edges,events,boundary_,stage_,begin,error);
    partials.insert(partials.end(),g.parameters.begin(),g.parameters.end());
    services.end(worker);
  }
  const auto registry=s.model.parameters(),all_parameters=s.model.parameters(false);
  std::vector<WeightedParameterPartial> trainable;
  for(const auto& contribution:partials){
    // Validate every role, including frozen parameters, before selecting the
    // requested trainable owners. A typo must not silently drop a gradient.
    const auto owner=all_parameters.value(contribution.name);
    if(owner.sizes()!=contribution.values.sizes())throw std::invalid_argument("weighted window role shape differs from owner");
    if(registry.contains(contribution.name))trainable.push_back(contribution);
  }
  auto part=append_weighted_parameter_vjp(p,registry,trainable,error,remaining(p));
  parameters_={part.owners,part.offsets,at::zeros_like(part.values),at::zeros_like(part.connected)};
  auto accumulated=append_private_parameter_accumulate(p,parameters_,part,error,remaining(p));
  p.copy(parameters_.connected,accumulated.connected);p.branch(index,{test});
  p.mark(init);p.zero(events.values);p.zero(events.connected);p.zero(boundary_.values);p.zero(boundary_.connected);
  p.zero(parameters_.values);p.zero(parameters_.connected);p.copy(stage_,s.stages);
  for(int64_t v=0;v<count;++v){auto& worker=services.begin();worker.zero(rolling[v].value);worker.zero(rolling[v].connected);
    weighted_reverse::add(worker,nodes_[v]->layout,rolling[v],roots_.nodes[v].final,error);services.end(worker);}
  weighted_window::route(p,s.pending->atoms(),roots_.pending,edges,events,boundary_,stage_,begin,error);p.branch(index,{test});
  p.mark(done);
  for(int64_t v=0;v<count;++v){auto& worker=services.begin();left_.push_back(weighted_window::mask(worker,nodes_[v]->layout,rolling[v],journal.boundary("node."+std::to_string(v)+".present"),error));services.end(worker);}
  services.append_stop();remaining(p);services.finish();p.finish();remaining(p);
}
}
