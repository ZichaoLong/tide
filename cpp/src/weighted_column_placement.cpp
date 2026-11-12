// Temporary Read/control views. Raw state, parameter identity and full B/L axes persist.
#include "weighted_column_placement.h"
#include "weighted_placement.h"
namespace tide::weighted::columns {
namespace {
bool read_parameter(const std::string& name,const MemoryConfiguration& config){
  if(name.rfind("read_program.",0)==0||name=="kernel.summary"||name.rfind("kernel.read_",0)==0)return true;
  return std::visit([&](const auto& c){
    if constexpr(std::is_same_v<std::decay_t<decltype(c)>,KVConfig>)return false;
    else return !c.basic()&&c.memory_read=="state"&&(name=="kernel.projection"||name=="kernel.delta_bias"||name.rfind("kernel.conv.",0)==0);
  },config);
}
}
at::ScalarType read_dtype(const ResolvedPlacement& p,at::ScalarType payload){
  auto dtype=p.scoring_dtype=="float32"?at::kFloat:p.scoring_dtype=="float64"?at::kDouble:payload;
  detail::require(dtype==at::kFloat||dtype==at::kDouble,"column Read requires FP32/FP64");
  if(dtype==at::kDouble)for(const auto& d:{p.read,p.control,p.selection})
    detail::require(d.type()!=c10::DeviceType::PrivateUse1,"NPU Read/control/selection cannot consume FP64 columns");
  return dtype;
}
namespace {
void place_read(NodeWeights& weights,const NodeBinding& binding,LocalWindow& query,
                const std::string& mode,LocalState& old,const ResolvedPlacement& placement){
  const auto dtype=read_dtype(placement,weights.bias.scalar_type());
  if(weights.bias.device()==placement.read&&weights.bias.scalar_type()==dtype)return;
  std::vector<Tensor*> handles;
  for(auto& [name,t]:weights.extra)if(read_parameter(name,binding.memory))handles.push_back(&t);
  std::visit([&](auto& q){
    handles.insert(handles.end(),{&q.mass,&q.content});
    if(mode!="content"){
      for(auto& [name,t]:q.bank){
        if constexpr(std::is_same_v<std::decay_t<decltype(q)>,KVWindow>){
          const bool sum=std::get<KVConfig>(binding.memory).memory_read=="sum";
          if(name=="weights"||(sum?name=="summaries":name=="keys"||name=="values"))handles.push_back(&t);
        }else handles.push_back(&t);
      }
      if constexpr(std::is_same_v<std::decay_t<decltype(q)>,RecurrentWindow>)handles.push_back(&q.cache_values);
    }
  },query);
  if(mode=="content")std::visit([&](auto& s){
    for(auto& [name,t]:s.fields)handles.push_back(&t);
    if constexpr(std::is_same_v<std::decay_t<decltype(s)>,RecurrentState>)handles.push_back(&s.cache_values);
  },old);
  placement_tensors(handles,placement.read,dtype);
}
}
TensorFields read_placed(const NodeWeights& original,const NodeBinding& binding,const LocalWindow& window,
                        const std::string& mode,const LocalState& initial,const ResolvedPlacement& placement){
  auto weights=original;auto query=window;auto old=initial;
  place_read(weights,binding,query,mode,old,placement);
  return read_columns(weights,binding,query,mode,old);
}
std::vector<TensorFields> read_placed_many(const std::vector<PlacedReadRequest>& requests,Index budget){
  std::vector<NodeWeights> weights;std::vector<LocalReadRequest> queries;weights.reserve(requests.size());queries.reserve(requests.size());
  for(const auto& r:requests){detail::require(r.query.weights,"placed Read requires parameter owner");
    weights.push_back(*r.query.weights);queries.push_back(r.query);auto& q=queries.back();q.weights=&weights.back();
    place_read(weights.back(),q.binding,q.window,q.mode,q.initial,r.placement);}
  return read_columns_many(queries,budget);
}
RegionColumns region_placed(const RegionWeights& original,const TensorFields& descriptors,const RegionalConfig& config,
                            const Tensor& nodes,const ResolvedPlacement& placement){
  const auto& ref=descriptors.at("w");
  bool copy=ref.device()!=placement.control;
  for(const auto& [name,t]:original.extra)copy|=t.device()!=placement.control||t.scalar_type()!=ref.scalar_type();
  if(!copy)return region_columns(original,descriptors,config,nodes,placement.selection);
  auto weights=original;auto fields=descriptors;
  const auto dtype=fields.at("w").scalar_type();std::vector<Tensor*> handles;
  for(auto& [name,t]:weights.extra)handles.push_back(&t);
  for(auto& [name,t]:fields)handles.push_back(&t);
  placement_tensors(handles,placement.control,dtype);
  return region_columns(weights,fields,config,nodes.to(placement.control),placement.selection);
}
Tensor control_payload(const Tensor& value,const Tensor& like){
  auto result=placement_copy(value,like.device(),like.scalar_type());
  if(value.scalar_type()!=result.scalar_type())
    detail::require(at::equal((value.detach()>0).to(at::kCPU),(result.detach()>0).to(at::kCPU)),"column control cast changes positive support");
  return result;
}
}
