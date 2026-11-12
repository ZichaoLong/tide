// Allocation §2–6: independent node-column roots through candidate-count groups.
#include "weighted_regional_groups.h"
#include "weighted_column_placement.h"
#include "tide/autograd_census.h"
#include "tide/operator_profile.h"
#include <ATen/record_function.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/saved_variable.h>
#include <set>

namespace tide::weighted::columns {
namespace {
using torch::autograd::variable_list;
using Saved=torch::autograd::SavedVariable;
struct GroupTape {
  std::vector<std::string> names,dependencies;
  std::vector<Saved> leaves,numeric;
  std::string mode,regularizer;
  std::optional<double> alpha;
  bool allocation=false;
  RegionColumns restore() const {
    std::vector<Tensor> xs;for(const auto& value:numeric)xs.push_back(value.unpack());
    RegionColumns result{xs.at(0),std::nullopt,mode};
    if(allocation){
      AllocationColumns a{xs.at(7),xs.at(8),xs.at(9),xs.at(10),xs.at(11),xs.at(12),{},
                          regularizer,regularizer+"-exact-map-v1",alpha};
      std::copy_n(xs.begin()+1,6,a.inputs.begin());result.allocation=std::move(a);
    }
    return result;
  }
};
class GroupedRegionVjp final:public torch::autograd::Node {
 public:
  RegionalPacking packing;
  size_t weights=0;
  std::vector<std::pair<Index,std::string>> names;
  std::vector<Index> output_nodes;
  std::vector<Saved> originals,parameters;
  std::vector<GroupTape> tapes;
  explicit GroupedRegionVjp(const RegionalPacking& p):packing(p){}
  variable_list apply(variable_list&& bars) override {
    RECORD_FUNCTION("tide::stage::region_group_backward",std::vector<c10::IValue>());
    op_profile::Scope profile(op_profile::RegionAdjoint);
    if(at::GradMode::is_enabled())throw std::invalid_argument("regional node groups support first-order VJP only");
    if(originals.empty())throw std::runtime_error("regional group VJP graph released; use retain_graph");
    variable_list inputs;for(const auto& value:originals)inputs.push_back(value.unpack());
    RegionalFields sources;std::map<std::pair<Index,std::string>,size_t> input_ids;
    for(size_t i=0;i<names.size();++i){sources[names[i].first][names[i].second]=inputs[weights+i];input_ids[names[i]]=weights+i;}
    std::map<Index,Tensor> node_bars;
    for(size_t i=0;i<bars.size();++i)node_bars[output_nodes[i]]=bars[i];
    auto adjoints=packing.controls_pullback(node_bars,tapes.at(0).numeric.at(0).unpack());
    variable_list result(inputs.size());
    auto accumulate=[&](size_t i,const Tensor& value){if(value.defined())result[i]=result[i].defined()?result[i]+value:value;};
    for(size_t group_id=0;group_id<tapes.size();++group_id){
      const auto& bar=adjoints.values[group_id];if(!bar.defined())continue;
      const auto& tape=tapes[group_id];const auto& group=packing.groups[group_id];
      auto column=tape.restore();variable_list leaves;
      for(const auto& parameter:parameters)leaves.push_back(parameter.unpack());
      for(const auto& leaf:tape.leaves)leaves.push_back(leaf.unpack());
      std::vector<size_t> leaf_ids;variable_list differentiable;
      for(size_t i=0;i<leaves.size();++i)if(leaves[i].requires_grad()){leaf_ids.push_back(i);differentiable.push_back(leaves[i]);}
      if(differentiable.empty())continue;
      auto support=adjoints.row_support[group_id].to(at::kCPU).contiguous();auto present=support.const_data_ptr<bool>();
      std::map<std::string,std::vector<Index>> branches;
      for(Index row=0;row<support.numel();++row)if(present[row])branches[tape.dependencies[row]].push_back(row);
      for(const auto& [dependency,rows]:branches){
        auto ids=at::tensor(rows,at::TensorOptions().dtype(at::kLong)).to(bar.device());
        auto active=at::zeros({support.numel()},bar.options().dtype(at::kBool)).index_fill(0,ids,true);
        auto incoming=at::where(active.unsqueeze(1),bar,0);
        auto roots=region_column_vjp_sources(column,incoming,active);if(roots.empty())continue;
        variable_list outputs,grad_outputs;for(const auto& [value,gradient]:roots){outputs.push_back(value);grad_outputs.push_back(gradient);}
        variable_list grads;{
          RECORD_FUNCTION("tide::stage::region_gradient",std::vector<c10::IValue>());
          op_profile::Scope gradient_profile(op_profile::RegionGradient);
          grads=torch::autograd::grad(outputs,differentiable,grad_outputs,true,false,true);
        }
        TensorFields field_bars;
        for(size_t j=0;j<leaf_ids.size();++j){auto i=leaf_ids[j];
          if(i<weights)accumulate(i,grads[j]);else field_bars[tape.names[i-weights]]=grads[j];}
        for(const auto& [node,fields]:group.pullback(sources,field_bars,active))
          for(const auto& [name,value]:fields)accumulate(input_ids.at({node,name}),value);
      }
    }
    for(size_t i=0;i<result.size();++i)if(!should_compute_output(i))result[i]=Tensor{};
    return result;
  }
  void release_variables() override {originals.clear();parameters.clear();tapes.clear();}
  std::string name() const override{return "tide::weighted::RegionalNodeGroupsVjp";}
};
}

std::map<Index,Tensor> region_groups(const RegionWeights& original,const RegionalConfig& config,
    const RegionalPacking& packing,const RegionalFields& fields,const ResolvedPlacement& placement){
  if(packing.groups.empty())return {};
  bool needs_grad=false,same_members=true;
  for(const auto& [name,value]:original.extra)needs_grad|=value.requires_grad();
  for(const auto& [node,domain]:packing.domains)for(const auto& [name,value]:fields.at(node))needs_grad|=value.requires_grad();
  for(const auto& group:packing.groups)same_members&=group.participants.size()==static_cast<size_t>(group.candidates);
  if(!at::GradMode::is_enabled()||!needs_grad||same_members){
    std::vector<Tensor> values;RegionParameterViews placed(original,placement.control);
    for(const auto& group:packing.groups){auto p=group.pack(fields);
      values.push_back(placed.evaluate(p.fields,config,p.nodes,placement.selection).controls);}
    return packing.unpack(values);
  }
  RECORD_FUNCTION("tide::stage::region_group_bind",std::vector<c10::IValue>());
  op_profile::Scope profile(op_profile::RegionBind);
  auto node=std::make_shared<GroupedRegionVjp>(packing);auto weights=original;
  variable_list inputs,parameter_leaves;std::map<const c10::TensorImpl*,Tensor> aliases;
  for(auto& [name,value]:weights.extra){
    const auto id=value.unsafeGetTensorImpl();auto found=aliases.find(id);
    if(found!=aliases.end())value=found->second;
    else{inputs.push_back(value);value=value.detach().set_requires_grad(value.requires_grad());
      aliases.emplace(id,value);parameter_leaves.push_back(value);node->parameters.emplace_back(value,false);}
  }
  node->weights=inputs.size();
  for(const auto& [id,domain]:packing.domains)for(const auto& [name,value]:fields.at(id)){
    node->names.emplace_back(id,name);inputs.push_back(value);}
  node->set_next_edges(torch::autograd::collect_next_edges(inputs));
  for(const auto& value:inputs)node->originals.emplace_back(value,false);
  std::vector<Tensor> controls;std::set<Index> enabled;
  RegionParameterViews placed(weights,placement.control);
  for(const auto& group:packing.groups){
    PackedRegionalColumns packed;{at::NoGradGuard guard;packed=group.pack(fields);}
    GroupTape tape;TensorFields leaves;
    for(auto& [name,value]:packed.fields){
      bool required=false;for(auto id:group.participants)required|=fields.at(id).at(name).requires_grad();
      auto leaf=value.detach().set_requires_grad(required);leaves[name]=leaf;
      tape.names.push_back(name);tape.leaves.emplace_back(leaf,false);
    }
    RegionColumns result;{at::AutoGradMode enable(true);result=placed.evaluate(leaves,config,packed.nodes,placement.selection);}
    tape.mode=result.mode;tape.dependencies=region_column_dependencies(result);
    std::vector<Tensor> saved{result.controls};
    if(result.allocation){const auto& a=*result.allocation;tape.allocation=true;tape.regularizer=a.regularizer;tape.alpha=a.alpha;
      saved.insert(saved.end(),a.inputs.begin(),a.inputs.end());
      for(const auto& value:{a.controls,a.multiplier,a.branch,a.response,a.free,a.residual})saved.push_back(value);}
    for(const auto& value:saved)tape.numeric.emplace_back(value,false);
    const auto differentiable=region_column_differentiable(result);
    for(size_t i=0;i<group.node_ids.size();++i)if(differentiable[i/group.candidates])enabled.insert(group.node_ids[i]);
    if(autograd_census::enabled()){
      autograd_census::graph("private/regional-groups",{result.controls});
      autograd_census::saved(saved);for(const auto& [name,value]:leaves)autograd_census::saved({value});}
    controls.push_back(result.controls);node->tapes.push_back(std::move(tape));
  }
  if(autograd_census::enabled()){autograd_census::saved(inputs);autograd_census::saved(parameter_leaves);}
  profile.phase(op_profile::RegionPublish);
  std::map<Index,Tensor> outputs;{at::NoGradGuard guard;outputs=packing.unpack(controls);}
  for(auto& [id,value]:outputs){value=value.detach();if(enabled.count(id)){
    node->output_nodes.push_back(id);torch::autograd::set_history(value,node);}}
  return outputs;
}
}
