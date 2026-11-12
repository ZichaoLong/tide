#include "block_settle_session_internal.h"
#include "block_kv_local_saved.h"
#include <ATen/core/grad_mode.h>
#include <iomanip>
#include <limits>
#include <sstream>
namespace tide::device_online::block_session_detail {
void remember(std::vector<Version>& out,const at::Tensor& x){if(x.defined())out.push_back({x,x._version(),x.const_data_ptr(),x.requires_grad()});}
void validate(const std::vector<Version>& items){for(const auto& x:items)
  if(x.value._version()!=x.version||x.value.const_data_ptr()!=x.data||x.value.requires_grad()!=x.trainable)
    throw std::logic_error("resident block parameter, retained window or gradient was modified");}
std::vector<at::Tensor> tensors(const BlockSettleFlowTape& t){
  std::vector<at::Tensor> out{t.positions,t.nodes.payload,t.nodes.present,t.nodes.times,t.edges.payload,t.edges.present,t.edges.times,
    t.outputs.payload,t.outputs.present,t.outputs.times,t.dispatcher->journal(),t.dispatcher->ledger()};
  for(const auto& r:t.regions){auto saved=block_local_detail::saved(*r.local);out.insert(out.end(),saved.begin(),saved.end());
    out.insert(out.end(),{r.nodes,r.atoms,r.atom_counts,r.physical_atoms,r.source_mapping});}return out;
}
ParameterVjp parameter_layout(const ParameterRegistry& registry,at::Device device,int64_t budget){
  ParameterVjp out;out.owners=registry.owners();int64_t count=0;
  for(const auto& p:out.owners){if(!p.value.device().is_cpu()||p.value.scalar_type()!=at::kFloat||p.value.numel()<1||
      p.value.numel()>std::numeric_limits<int64_t>::max()-count)throw std::invalid_argument("block canonical registry requires finite CPU FP32 owners");
    out.offsets.push_back(count);count+=p.value.numel();}
  const auto size=std::max<int64_t>(1,count),owners=std::max<int64_t>(1,out.owners.size());
  if(4.L*size+owners>budget)throw std::invalid_argument("block canonical parameter layout exceeds budget");
  out.values=at::zeros({size},at::TensorOptions().device(device).dtype(at::kFloat));out.connected=at::zeros({owners},out.values.options().dtype(at::kBool));return out;
}
namespace {
template<class T> void emit(std::ostream& o,const T& x){o<<x<<' ';}
void emit(std::ostream& o,const std::string& x){o<<std::quoted(x)<<' ';}
template<class T> void emit(std::ostream& o,const std::optional<T>& x){o<<bool(x)<<' ';if(x)emit(o,*x);}
template<class T> void emit(std::ostream& o,const std::vector<T>& xs){o<<xs.size()<<' ';for(const auto& x:xs)emit(o,x);}
}
std::string identity(const SettleGraph& g,const std::vector<BlockSettleKVBinding>& bindings,const ParameterRegistry& registry,
    const std::vector<BlockKVParameterRoles>& roles,const std::vector<WeightedRawState>& state){
  std::ostringstream o;o<<std::setprecision(17)<<"tide-resident-settle-block-v1 ";emit(o,g.graph().identity);emit(o,g.ranks());emit(o,g.stride());emit(o,g.binding());
  for(size_t r=0;r<bindings.size();++r){const auto& b=bindings[r];const auto& c=b.memory;const auto& s=b.regional;
    for(auto x:{c.width,c.summary_width,c.adapter_rank,c.heads,c.ffn_width,c.window})emit(o,x);emit(o,c.memory_read);
    for(auto x:{c.beta_short,c.epsilon,c.alpha_att,c.alpha_ff,c.position_origin,c.position_scale})emit(o,x);
    emit(o,c.full_frequencies);emit(o,c.read_frequencies);emit(o,c.distance_slopes);emit(o,b.read_mode);emit(o,b.aggregate_alpha);
    emit(o,s.hidden_width);emit(o,s.region_control_width);emit(o,s.k);emit(o,s.gamma_max);
    for(const auto& x:{s.allocation_temperature,s.total_fraction,s.ent_alpha,s.eta,s.fixed_gain,s.gain_logit,s.alpha_match,s.alpha_res,s.alpha_loc,s.alpha_sel})emit(o,x);
    emit(o,s.selection_mode);emit(o,s.region_control_mode);emit(o,s.region_evidence);emit(o,s.regularizer);emit(o,s.send_gain);emit(o,s.allocation_base);
    emit(o,s.local_score);emit(o,s.region_score);emit(o,s.selection_width);emit(o,s.region_key);emit(o,s.query_weight);
    emit(o,b.node_owners);emit(o,b.input_owners);emit(o,b.edge_owners);
    for(const auto* x:{&b.parameters.message.read.region.context_owners,&b.parameters.message.read.region.query_owners}){
      emit(o,x->defined());if(x->defined()){const auto cpu=x->cpu().contiguous();for(int64_t j=0;j<cpu.numel();++j)emit(o,cpu.reshape({-1})[j].item<int64_t>());}}
    for(const auto& [name,paths]:roles[r]){emit(o,name);emit(o,paths);}emit(o,state[r].floating.sizes().vec());emit(o,state[r].integer.sizes().vec());
  }
  emit(o,registry.alias_partitions());for(const auto& p:registry.owners()){emit(o,p.canonical);emit(o,p.value.sizes().vec());}
  return o.str();
}
}
