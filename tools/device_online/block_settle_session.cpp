#include "block_settle_session_internal.h"
#include "block_parameter_views.h"
#include <ATen/core/grad_mode.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <algorithm>
#include <limits>
namespace tide::device_online {
using namespace block_session_detail;
BlockSettleSession::Impl::Impl(SettleGraph g,std::vector<BlockSettleKVBinding> b,const ParameterRegistry& original,
    std::vector<BlockKVParameterRoles> names,std::vector<WeightedRawState> initial,std::vector<WeightedRawState> bound_initial,
    at::Device d,DeviceOptimizerKind k,std::vector<OptimizerGroup> groups,BlockSettleLimits l,const BlockSettleCheckpoint* saved,
    const BlockSettleBoundary* initial_metadata,const at::Tensor& initial_positions)
    :graph(std::move(g)),bindings(std::move(b)),roles(std::move(names)),device(d),limits(l),kind(k){
  if(at::GradMode::is_enabled()||d.type()!=resident_device_type||d.index()<0||l.forward_bytes<1||l.backward_bytes<1||l.optimizer_bytes<1||
      l.total_bytes<1||l.workspace_bytes<1||l.windows<1||l.timeout_ms<1||bindings.empty()||roles.size()!=bindings.size()||
      initial.size()!=bindings.size()||bound_initial.size()!=bindings.size())throw std::invalid_argument("invalid resident block owner binding/capacity");
  validate_kernel_device(d);for(const auto& p:original.owners()){
    if(!p.value.device().is_cpu()||p.value.scalar_type()!=at::kFloat||!at::isfinite(p.value).all().item<bool>())
      throw std::invalid_argument("block owner requires finite canonical CPU FP32 parameters");
    remember(originals,p.value);auto x=p.value.detach().clone();for(const auto& alias:p.aliases)registry.add(alias,x);}
  fingerprint=identity(graph,bindings,original,roles,initial);
  if(saved){if(saved->schema!=2||saved->identity!=fingerprint||saved->aliases!=registry.alias_partitions()||saved->generation<0||saved->next_token<0||
      saved->parameters.size()!=registry.owners().size()||saved->state.size()!=initial.size()||saved->bound_initial.size()!=initial.size())
      throw std::invalid_argument("block checkpoint semantic/owner identity differs");
    for(const auto& p:registry.owners()){const auto& x=saved->parameters.at(p.canonical);if(!x.defined()||!x.device().is_cpu()||x.scalar_type()!=at::kFloat||
        x.sizes()!=p.value.sizes()||x.requires_grad()||!x.is_contiguous()||!at::isfinite(x).all().item<bool>())throw std::invalid_argument("invalid block checkpoint parameter");}
    for(const auto& p:registry.owners())p.value.copy_(saved->parameters.at(p.canonical));
    initial=saved->state;bound_initial=saved->bound_initial;kind=saved->optimizer;groups=saved->groups;generation=saved->generation;next_token=saved->next_token;}
  for(size_t r=0;r<bindings.size();++r){auto& binding=bindings[r];const auto count=int64_t(binding.node_owners.size());
    if(count<1||!initial[r].floating.defined()||initial[r].floating.dim()!=2||initial[r].floating.size(0)%count)
      throw std::invalid_argument("block initial rows do not match private nodes");
    WeightedRawLayout raw(binding.memory,initial[r].integer.size(1)-4);validate_weighted_raw_buffers(raw,initial[r]);validate_weighted_raw_buffers(raw,bound_initial[r]);
    if(!initial[r].floating.device().is_cpu()||!bound_initial[r].floating.device().is_cpu()||initial[r].floating.scalar_type()!=at::kFloat||
        bound_initial[r].floating.sizes()!=initial[r].floating.sizes()||bound_initial[r].integer.sizes()!=initial[r].integer.sizes())
      throw std::invalid_argument("block initial/raw Read0 binding differs");
    decode_weighted_raw_state(raw,initial[r]);decode_weighted_raw_state(raw,bound_initial[r]);
    const auto batch=initial[r].floating.size(0)/count;if(!positions.defined())positions=at::full({batch},-1,at::kLong);
    if(positions.numel()!=batch)throw std::invalid_argument("block region sample count differs");
    state.push_back(copy_weighted_raw_state(initial[r],d));bound.push_back(copy_weighted_raw_state(bound_initial[r],d));
    const auto fields=block_kv_local_parameters(binding.parameters);if(fields.size()!=roles[r].size())throw std::invalid_argument("block owner requires complete role mapping");
    std::map<std::string,at::Tensor> packed;
    for(const auto& [role,bank]:fields){const auto& paths=roles[r].at(role);if(bank.dim()<1||paths.size()!=size_t(bank.size(0)))throw std::invalid_argument("block role bank row mapping differs");
      std::vector<at::Tensor> rows;for(size_t j=0;j<paths.size();++j){const auto x=registry.value(paths[j]);
        auto packed=block_parameter_pack(role,x,bank[j].sizes());
        if(!saved&&!at::equal(packed,bank[j].cpu()))throw std::invalid_argument("canonical block role shape/value differs");rows.push_back(packed);}
      packed[role]=at::stack(rows).to(d);}
    binding.parameters=block_kv_local_parameters(packed,binding.parameters);
    for(auto* x:{&binding.parameters.message.read.region.context_owners,&binding.parameters.message.read.region.query_owners})if(x->defined())*x=x->detach().to(d).clone();
  }
  const auto batch=positions.numel(),nodes=int64_t(graph.graph().nodes.size()),regions=int64_t(bindings.size()),ports=int64_t(graph.graph().inputs.size());
  BlockSettleBoundary metadata{at::ones({batch,nodes},at::kBool),at::full({batch,regions},-1,at::kLong),at::full({batch,ports},-1,at::kLong),at::full({batch,ports},-1,at::kLong)};
  if(initial_metadata)metadata=*initial_metadata;if(saved)metadata=saved->boundary;
  auto bound_meta=[&](const at::Tensor& x,at::IntArrayRef shape,at::ScalarType type){
    if(!x.defined()||!x.device().is_cpu()||x.scalar_type()!=type||x.sizes()!=shape||!x.is_contiguous()||x.requires_grad())throw std::invalid_argument("block complete-cut metadata layout differs");
    if(type==at::kLong&&x.lt(-1).any().item<bool>())throw std::invalid_argument("invalid block complete-cut clock/count");return x.to(d).clone();};
  boundary={bound_meta(metadata.node_present,{batch,nodes},at::kBool),bound_meta(metadata.region_times,{batch,regions},at::kLong),
    bound_meta(metadata.input_counts,{batch,ports},at::kLong),bound_meta(metadata.input_times,{batch,ports},at::kLong)};
  if(!at::equal(metadata.input_counts.eq(-1),metadata.input_times.eq(-1)))throw std::invalid_argument("block input occurrence/time presence differs");
  if(initial_positions.defined()){
    if(!initial_positions.device().is_cpu()||initial_positions.scalar_type()!=at::kLong||initial_positions.sizes()!=positions.sizes()||initial_positions.lt(-1).any().item<bool>())
      throw std::invalid_argument("initial block sample cut differs");positions=initial_positions.detach().clone();}
  layout=block_session_detail::parameter_layout(registry);
  optimizer=std::make_unique<DeviceOptimizer>(layout,d,kind,std::move(groups),std::min(l.optimizer_bytes,l.total_bytes));
  if(saved){if(!saved->last_positions.defined()||!saved->last_positions.device().is_cpu()||saved->last_positions.scalar_type()!=at::kLong||
      saved->last_positions.sizes()!=positions.sizes()||saved->last_positions.lt(-1).any().item<bool>())throw std::invalid_argument("invalid block checkpoint sample cut");
    if(saved->slots.values.sizes()!=optimizer->values().sizes())throw std::invalid_argument("block checkpoint canonical extent differs");
    for(size_t j=0;j<layout.owners.size();++j)if(!at::equal(saved->slots.values.narrow(0,layout.offsets[j],layout.owners[j].value.numel()).reshape(layout.owners[j].value.sizes()),layout.owners[j].value))
      throw std::invalid_argument("block checkpoint named/master values disagree");
    optimizer->restore(saved->slots);positions=saved->last_positions.clone();}
  if(saved||initial_positions.defined())for(int64_t b=0;b<batch;++b){const auto position=positions[b].item<int64_t>();
    if(position>=std::numeric_limits<int64_t>::max()/graph.stride())throw std::invalid_argument("block checkpoint complete cut overflows int64");
    const auto cut=(position+1)*graph.stride();
    for(int64_t r=0;r<regions;++r){const auto time=metadata.region_times[b][r].item<int64_t>();
      if(time>=cut||(time>=0&&time%graph.stride()!=graph.ranks()[r]))throw std::invalid_argument("block checkpoint regional history outside cut/time code");
      const auto count=int64_t(bindings[r].node_owners.size());
      if(initial[r].integer.narrow(0,b*count,count).select(1,0).ge(cut).any().item<bool>())throw std::invalid_argument("block checkpoint raw state outside cut");}
    for(int64_t port=0;port<ports;++port){const auto time=metadata.input_times[b][port].item<int64_t>();
      if(time>=cut||(time>=0&&time%graph.stride()!=graph.rank(graph.graph().inputs[port])))throw std::invalid_argument("block checkpoint input ledger outside cut/time code");}}
  budget();
}
void BlockSettleSession::Impl::check()const{
  if(at::GradMode::is_enabled()||closed||failed)throw std::logic_error("resident block owner closed/failed or grad enabled");
  validate(originals);validate(gradient_versions);for(const auto& w:windows)validate(w.versions);
}
void BlockSettleSession::Impl::run(DeviceProgram& p,const at::Tensor& error){
  c10::impl::VirtualGuardImpl(device.type()).synchronizeDevice(device.index());p.run(limits.timeout_ms);
  const auto code=error.cpu().item<int>();if(code)throw std::runtime_error("resident block owner refused, device code="+std::to_string(code));
}
void BlockSettleSession::Impl::discard(){
  for(auto& w:windows)w.program->close();windows.clear();if(reverse){reverse->close();reverse.reset();}gradient={};gradient_versions.clear();ready=false;
}
BlockSettleSession::BlockSettleSession(SettleGraph g,std::vector<BlockSettleKVBinding> b,const ParameterRegistry& p,
    std::vector<BlockKVParameterRoles> names,std::vector<WeightedRawState> state,std::vector<WeightedRawState> bound,
    at::Device d,DeviceOptimizerKind k,std::vector<OptimizerGroup> groups,BlockSettleLimits l,const BlockSettleCheckpoint* c,
    const BlockSettleBoundary* metadata,const at::Tensor& positions)
    :impl_(std::make_unique<Impl>(std::move(g),std::move(b),p,std::move(names),std::move(state),std::move(bound),d,k,std::move(groups),l,c,metadata,positions)){}
BlockSettleSession::~BlockSettleSession()=default;
BlockSettleWindow BlockSettleSession::advance(const at::Tensor& positions,const at::Tensor& payload,const at::Tensor& present){
  auto& s=*impl_;s.check();if(s.ready||s.windows.size()>=size_t(s.limits.windows)||s.next_token==std::numeric_limits<int64_t>::max())
    throw std::invalid_argument("consume block gradients or detach before retained window capacity");
  if(!positions.defined()||positions.scalar_type()!=at::kLong||positions.dim()!=2||positions.size(0)!=s.positions.numel()||positions.size(1)<1)
    throw std::invalid_argument("block positions require int64 B,L");
  const auto host=positions.detach().cpu();if(host.lt(0).any().item<bool>()||host.select(1,0).le(s.positions).any().item<bool>()||
      (host.size(1)>1&&host.slice(1,1).le(host.slice(1,0,-1)).any().item<bool>()))throw std::invalid_argument("block windows must advance original sample positions");
  if(!payload.defined()||!present.defined())throw std::invalid_argument("block window requires payload and presence tensors");
  const auto input_bytes=resident_bytes_add(positions.nbytes(),resident_bytes_add(payload.nbytes(),present.nbytes()));
  require_resident_bytes(input_bytes,s.limits.forward_bytes,"forward input tensor");
  const auto live=s.storage();require_resident_bytes(resident_bytes_add(resident_bytes_add(live.tensors,live.workspaces),input_bytes),s.limits.total_bytes,"total before input copy");
  BlockSettleFlowInput input{positions.detach().to(s.device).clone(),payload.detach().to(s.device).clone(),present.detach().to(s.device).clone(),s.state,s.bound};
  auto program=std::make_unique<DeviceProgram>(s.device);program->limit_workspace(s.limits.workspace_bytes);
  const auto before=s.storage().tensors;
  auto tape=std::make_shared<BlockSettleFlowTape>(append_block_settle_flow(*program,s.graph,s.bindings,input,s.limits.forward_bytes));
  auto boundary=append_block_settle_boundary(*program,s.graph,*tape,input.present,s.boundary,s.limits.forward_bytes);
  s.finish_and_admit(*program,before,s.limits.forward_bytes,"forward tensor");
  try{s.run(*program,tape->dispatcher->error());}catch(...){s.failed=true;throw;}
  BlockSettleWindow view{s.next_token++,s.generation,tape,boundary};Impl::Window saved{view,std::move(program),{}};
  for(const auto& x:tensors(*tape))remember(saved.versions,x);
  for(const auto& x:{boundary.node_present,boundary.region_times,boundary.input_counts,boundary.input_times})remember(saved.versions,x);
  for(size_t r=0;r<s.state.size();++r)s.state[r]=tape->regions[r].local->sequence.final;
  s.boundary=std::move(boundary);
  s.positions=host.select(1,host.size(1)-1).clone();s.windows.push_back(std::move(saved));return view;
}
void BlockSettleSession::detach(){auto& s=*impl_;s.check();s.discard();}
void BlockSettleSession::close(){if(!impl_||impl_->closed)return;impl_->discard();impl_->optimizer.reset();impl_->closed=true;}
}
