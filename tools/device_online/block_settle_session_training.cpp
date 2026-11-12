#include "block_settle_session_internal.h"
#include "block_read_selection_internal.h"
#include "block_parameter_views.h"
#include "sharded_parameter_banks.h"
#include "device_launch_tide_block_raw_roots.h"
#include <limits>
#include <set>
namespace tide::device_online {
namespace {
void carry(DeviceProgram& p,const BlockKVSequenceTape& previous,const BlockKVSequenceGradient& next,
    BlockKVSequenceRoots& roots,const at::Tensor& error){
  using namespace block_selection_detail;const auto rows=previous.final.floating.size(0),width=previous.layout.floating_width();
  const auto count=int64_t(previous.layout.fields().size()),capacity=previous.layout.capacity();
  if(roots.final.defined()!=roots.final_connected.defined())throw std::invalid_argument("cross-window final root requires value and fields");
  auto value=at::zeros_like(previous.final.floating),flags=at::zeros({rows,count},error.options().dtype(at::kBool));p.zero(value);p.zero(flags);
  const auto mapping=at::arange(rows,previous.fields.options()),metadata=previous.final.integer,fields=previous.fields;
  for(const auto& source:std::vector<WeightedRowAdjoint>{{roots.final,roots.final_connected},{next.initial,next.initial_connected}}){
    if(!source.value.defined())continue;check(source.value,error,at::kFloat,{rows,width});check(source.connected,error,at::kBool,{rows,count});
    p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_block_raw_roots)(1,stream,
      ptr(fields),ptr(metadata),ptr(mapping),ptr(source.value),ptr(source.connected),ptr(value),ptr(flags),ptr(error),rows,rows,width,capacity,count),
      "exact raw state adjoint across retained block windows");},{fields,metadata,mapping,source.value,source.connected,value,flags,error});
  }
  roots.final=value;roots.final_connected=flags;
}
}
BlockSettleAdjoints BlockSettleSession::backward(const std::vector<BlockSettleFlowRoots>& inputs){
  auto& s=*impl_;s.check();if(s.ready||s.windows.empty()||inputs.size()!=s.windows.size())throw std::logic_error("block backward needs all retained windows exactly once");
  auto program=std::make_unique<DeviceProgram>(s.device);program->limit_workspace(s.limits.workspace_bytes);
  auto services=std::make_unique<TaskServices>(*program,s.device,s.limits.workspace_bytes);
  const auto before=s.storage().tensors;
  BlockSettleAdjoints out;out.windows.resize(inputs.size());auto error=at::zeros({1},s.optimizer->values().options().dtype(at::kInt));program->zero(error);
  for(int64_t w=int64_t(inputs.size())-1;w>=0;--w){auto roots=inputs[w];const auto& t=*s.windows[w].view.tape;
    if(roots.regions.empty())roots.regions.resize(t.regions.size());
    if(roots.regions.size()!=t.regions.size())throw std::invalid_argument("block retained root region count differs");
    program->add(t.dispatcher->error(),error);
    if(w+1<int64_t(inputs.size()))for(size_t r=0;r<t.regions.size();++r)
      carry(*program,t.regions[r].local->sequence,out.windows[w+1].regions[r].sequence,roots.regions[r].raw,t.dispatcher->error());
    out.windows[w]=append_block_settle_flow_vjp(*program,*services,s.graph,t,roots,s.limits.backward_bytes);
    program->copy(error,t.dispatcher->error());
  }
  std::vector<WeightedParameterPartial> parts;std::set<std::string> used;
  for(const auto& window:out.windows)for(size_t r=0;r<s.roles.size();++r){const auto found=block_kv_local_parameter_roots(window.regions[r]);
    for(const auto& [role,names]:s.roles[r]){
      if(!found.count(role)&&role.rfind("source.",0)==0&&!window.regions[r].aggregate.parameters)continue;
      const auto& root=found.at(role);if(!root.value.defined()||root.value.size(0)!=int64_t(names.size()))throw std::logic_error("block role gradient extent differs");
      for(size_t bank=0;bank<names.size();++bank){auto value=block_parameter_view(role,root.value[bank],s.registry.value(names[bank]).sizes());
        if(!value.is_contiguous()){
          // This is a view of a padded source bank produced only when reverse
          // runs. A host contiguous() would read it before computation; the
          // control copy also requires contiguous buffers. Gather the static
          // rectangular coordinates on the device after the producer.
          const auto bank_value=root.value[bank];
          if(!bank_value.is_contiguous())throw std::logic_error("block parameter bank row is not contiguous");
          auto coordinates=at::arange(bank_value.numel(),at::kLong).reshape(bank_value.sizes());
          auto indices=block_parameter_view(role,coordinates,value.sizes()).contiguous().reshape({-1}).to(value.device());
          auto contiguous=at::zeros(value.sizes(),value.options());
          program->index_select(bank_value.reshape({-1}),0,indices,contiguous.reshape({-1}));value=contiguous;
        }
        parts.push_back({names[bank],value,root.connected[bank]});used.insert(s.registry.canonical_name(names[bank]));}}
  }
  // Keep the optimizer's complete canonical layout. Unused owners have an
  // explicit false bit, which preserves None and prevents decay/step changes.
  auto absent=at::zeros({1},s.optimizer->values().options().dtype(at::kBool));
  for(const auto& owner:s.layout.owners)if(!used.count(owner.canonical))parts.push_back({owner.canonical,at::zeros(owner.value.sizes(),s.optimizer->values().options()),absent});
  out.parameters=append_weighted_parameter_vjp(*program,s.registry,parts,error,s.limits.backward_bytes);
  if(out.parameters.offsets!=s.layout.offsets)throw std::logic_error("block canonical reduction changed complete layout");
  for(size_t r=0;r<s.bindings.size();++r){const auto& t=s.windows.front().view.tape->regions[r].local->sequence;
    const auto rows=s.bound[r].floating.size(0),width=t.layout.floating_width(),fields_count=int64_t(t.layout.fields().size()),capacity=t.layout.capacity();
    WeightedRowAdjoint result{at::zeros_like(s.bound[r].floating),at::zeros({rows,fields_count},error.options().dtype(at::kBool))};
    program->zero(result.value);program->zero(result.connected);const auto mapping=at::arange(rows,t.fields.options()),fields=t.fields,metadata=s.bound[r].integer;
    for(const auto& window:out.windows){const auto value=window.regions[r].bound_initial,flags=window.regions[r].bound_initial_connected;
      program->kernel([=](void* stream){using block_selection_detail::ptr;check_device_launch(TIDE_LAUNCH_KERNEL(tide_block_raw_roots)(1,stream,
        ptr(fields),ptr(metadata),ptr(mapping),ptr(value),ptr(flags),ptr(result.value),ptr(result.connected),ptr(error),rows,rows,width,capacity,fields_count),
        "fixed Read0 initial binding accumulates every retained window");},{fields,metadata,mapping,value,flags,result.value,result.connected,error});}
    out.bound_initial.push_back(std::move(result));
  }
  s.finish_and_admit(*program,before,s.limits.backward_bytes,"backward tensor",0,services.get());
  try{s.run(*program,error,services.get());}catch(...){s.failed=true;throw;}
  s.gradient=out.parameters;s.reverse=std::move(program);s.reverse_services=std::move(services);s.ready=true;
  for(const auto& x:{s.gradient.values,s.gradient.connected})block_session_detail::remember(s.gradient_versions,x);
  return out;
}
void BlockSettleSession::Impl::publish(DeviceProgram& p,const at::Tensor& error){
  std::map<std::string,int64_t> offsets;for(size_t j=0;j<layout.owners.size();++j)offsets[layout.owners[j].canonical]=layout.offsets[j];
  const auto master=optimizer->values();std::vector<ParameterWrite> writes;
  for(size_t r=0;r<bindings.size();++r)for(const auto& [role,bank]:block_kv_local_parameters(bindings[r].parameters)){
    const auto& names=roles[r].at(role);for(size_t j=0;j<names.size();++j){const auto target=block_parameter_view(role,bank[j],registry.value(names[j]).sizes());
      writes.push_back({offsets.at(registry.canonical_name(names[j])),{target,at::kFloat}});}}
  append_owner_parameter_publish(p,writes,master,error,limits.optimizer_bytes);
}
void BlockSettleSession::step(){
  auto& s=*impl_;s.check();if(!s.ready||s.generation==std::numeric_limits<int64_t>::max())throw std::logic_error("block optimizer needs unconsumed gradient generation");
  DeviceProgram program(s.device);program.limit_workspace(s.limits.workspace_bytes);const auto before=s.storage().tensors;
  auto error=at::zeros({1},s.gradient.values.options().dtype(at::kInt));
  s.optimizer->append_step(program,s.gradient,error);
  auto zero=at::zeros_like(error),ok=at::zeros_like(error,error.options().dtype(at::kBool)),branch=at::zeros_like(error);
  const auto publish=program.label(),finish=program.label();program.equal(error,zero,ok);program.cast_index(ok,branch);program.branch(branch,{finish,publish});
  program.mark(publish);s.publish(program,error);program.mark(finish);
  s.finish_and_admit(program,before,s.limits.optimizer_bytes,"optimizer step tensor",s.optimizer->tensor_bytes());
  try{s.run(program,error);}catch(...){s.failed=true;throw;}program.close();s.discard();++s.generation;
}
BlockSettleCheckpoint BlockSettleSession::checkpoint()const{
  const auto& s=*impl_;s.check();if(!s.windows.empty()||s.ready)throw std::logic_error("block checkpoint requires a detached complete cut");
  BlockSettleCheckpoint out;out.identity=s.fingerprint;out.generation=s.generation;out.next_token=s.next_token;out.aliases=s.registry.alias_partitions();
  out.optimizer=s.kind;out.groups=s.optimizer->groups();out.slots=s.optimizer->snapshot();out.last_positions=s.positions.clone();
  out.boundary={s.boundary.node_present.cpu().clone(),s.boundary.region_times.cpu().clone(),
    s.boundary.input_counts.cpu().clone(),s.boundary.input_times.cpu().clone()};
  for(size_t j=0;j<s.layout.owners.size();++j){const auto& owner=s.layout.owners[j];
    out.parameters[owner.canonical]=out.slots.values.narrow(0,s.layout.offsets[j],owner.value.numel()).reshape(owner.value.sizes()).clone();}
  auto copy=[](const WeightedRawState& raw){return WeightedRawState{raw.floating.detach().cpu().clone(),raw.integer.detach().cpu().clone()};};
  for(const auto& raw:s.state)out.state.push_back(copy(raw));for(const auto& raw:s.bound)out.bound_initial.push_back(copy(raw));return out;
}
}
