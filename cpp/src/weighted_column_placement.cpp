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
std::vector<Tensor*> read_handles(NodeWeights& weights,const NodeBinding& binding,LocalWindow& query,
                                  const std::string& mode,LocalState& old){
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
  return handles;
}
void place_read(NodeWeights& weights,const NodeBinding& binding,LocalWindow& query,
                const std::string& mode,LocalState& old,const ResolvedPlacement& placement){
  const auto dtype=read_dtype(placement,weights.bias.scalar_type());
  if(weights.bias.device()==placement.read&&weights.bias.scalar_type()==dtype)return;
  placement_tensors(read_handles(weights,binding,query,mode,old),placement.read,dtype);
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
    weights.push_back(*r.query.weights);queries.push_back(r.query);queries.back().weights=&weights.back();}
  // One alias-preserving transport per destination; request-local copies would
  // destroy shared matrix identity before the tensor numerical boundary.
  using Key=std::tuple<int,int,int>;std::map<Key,std::vector<Tensor*>> groups;
  for(size_t i=0;i<requests.size();++i){const auto& p=requests[i].placement;auto& w=weights[i];auto& q=queries[i];
    const auto dtype=read_dtype(p,w.bias.scalar_type());
    if(w.bias.device()==p.read&&w.bias.scalar_type()==dtype)continue;
    auto handles=read_handles(w,q.binding,q.window,q.mode,q.initial);
    auto& group=groups[{int(p.read.type()),p.read.index(),int(dtype)}];group.insert(group.end(),handles.begin(),handles.end());}
  for(const auto& [key,handles]:groups)placement_tensors(handles,
    at::Device(static_cast<c10::DeviceType>(std::get<0>(key)),std::get<1>(key)),static_cast<at::ScalarType>(std::get<2>(key)));
  return read_columns_many(queries,budget);
}
const RegionWeights& RegionParameterViews::parameters(at::ScalarType dtype){
  if(auto found=views_.find(dtype);found!=views_.end())return *found->second;
  bool copy=false;for(const auto& [name,t]:original_.extra)
    copy|=t.device()!=control_||(t.is_floating_point()&&t.scalar_type()!=dtype);
  const RegionWeights* view=&original_;
  if(copy){auto& weights=owned_.emplace(dtype,original_).first->second;std::vector<Tensor*> handles;
    for(auto& [name,t]:weights.extra)handles.push_back(&t);
    placement_tensors(handles,control_,dtype);view=&weights;}
  views_[dtype]=view;return *view;
}
RegionColumns RegionParameterViews::evaluate(const TensorFields& descriptors,const RegionalConfig& config,
                                              const Tensor& nodes,at::Device selection){
  const auto& ref=descriptors.at("w");const auto& weights=parameters(ref.scalar_type());
  if(ref.device()==control_)return region_columns(weights,descriptors,config,nodes,selection);
  auto fields=descriptors;std::vector<Tensor*> handles;for(auto& [name,t]:fields)handles.push_back(&t);
  placement_tensors(handles,control_,ref.scalar_type());
  return region_columns(weights,fields,config,nodes.defined()?nodes.to(control_):nodes,selection);
}
RegionColumns region_placed(const RegionWeights& original,const TensorFields& descriptors,const RegionalConfig& config,
                            const Tensor& nodes,const ResolvedPlacement& placement){
  return RegionParameterViews(original,placement.control).evaluate(descriptors,config,nodes,placement.selection);
}
Tensor control_payload(const Tensor& value,const Tensor& like){
  auto result=placement_copy(value,like.device(),like.scalar_type());
  if(value.scalar_type()!=result.scalar_type())
    detail::require(at::equal((value.detach()>0).to(at::kCPU),(result.detach()>0).to(at::kCPU)),"column control cast changes positive support");
  return result;
}
}
