#include "weighted_private_rows.h"
#include "tide/operator_profile.h"
#include "tide/autograd_census.h"
#include <ATen/record_function.h>
// Allocation §2–6: independent frame/control roots on actual batch graphs.
#include "weighted_region_batch.h"
#include "tide/weighted_columns.h"
#include "tide/allocation_failure.h"
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/saved_variable.h>
#include <sstream>

namespace tide::weighted::batch {
namespace {
using torch::autograd::variable_list;
class RegionVjp final:public torch::autograd::Node {
 public:
  size_t weights=0,row_inputs=0,row_outputs=0,rows=0;
  private_rows::Packing packing;
  std::vector<torch::autograd::SavedVariable> originals,leaves,outputs;
  std::vector<std::string> dependencies;
  std::vector<size_t> output_map;
  bool allocation=false;
  std::string mode,regularizer;
  std::optional<double> alpha;
  variable_list apply(variable_list&& bars) override {
  RECORD_FUNCTION("tide::stage::region_backward", std::vector<c10::IValue>());
  op_profile::Scope stage_profile(op_profile::RegionAdjoint);
    if(at::GradMode::is_enabled())throw std::invalid_argument("batched region supports first-order VJP only");
    if(originals.empty())throw std::runtime_error("regional VJP graph already released; use retain_graph");
    for(auto& t:originals)t.unpack();variable_list xs,ys,full(rows*row_outputs);
    for(auto& t:leaves)xs.push_back(t.unpack());for(auto& t:outputs)ys.push_back(t.unpack());
    RegionColumns column{ys[0],std::nullopt,mode};
    if(allocation){
      AllocationColumns a{ys[7],ys[8],ys[9],ys[10],ys[11],ys[12],{},regularizer,regularizer+"-exact-map-v1",alpha};
      std::copy_n(ys.begin()+1,6,a.inputs.begin());column.allocation=std::move(a);
    }
    for(size_t i=0;i<bars.size();++i)full[output_map[i]]=bars[i];
    using Key=std::pair<std::vector<size_t>,std::string>;std::map<Key,std::vector<size_t>> groups;
    for(size_t row=0;row<rows;++row){
      std::vector<size_t> mask;for(size_t j=0;j<row_outputs;++j)if(full[row*row_outputs+j].defined())mask.push_back(j);
      if(!mask.empty())groups[{mask,dependencies[row]}].push_back(row);
    }
    variable_list result(weights+rows*row_inputs);
    for(const auto& [key,used]:groups){
      variable_list roots,grads,inputs,bars;std::vector<size_t> ids;std::vector<Index> flat,used_rows;
      for(auto row:used){used_rows.push_back(row);for(auto j:key.first){flat.push_back(row*row_outputs+j);bars.push_back(full[row*row_outputs+j]);}}
      auto indices=at::tensor(flat,column.controls.options().dtype(at::kLong));
      auto grad=at::zeros_like(column.controls).flatten().index_copy_(0,indices,at::stack(bars)).reshape(column.controls.sizes());
      auto support=at::zeros({Index(rows)},column.controls.options().dtype(at::kBool));
      support.index_fill_(0,at::tensor(used_rows,indices.options()),true);
      for(const auto& [root,bar]:region_column_vjp_sources(column,grad,support)){roots.push_back(root);grads.push_back(bar);}
      auto append=[&](size_t i){if(xs[i].requires_grad()){ids.push_back(i);inputs.push_back(xs[i]);}};
      for(size_t i=0;i<weights;++i)append(i);for(size_t f=0;f<row_inputs;++f)append(weights+f);
      if(inputs.empty()||roots.empty())continue;
      variable_list values;{
        RECORD_FUNCTION("tide::stage::region_gradient", std::vector<c10::IValue>());
        op_profile::Scope profile(op_profile::RegionGradient);
        values=torch::autograd::grad(roots,inputs,grads,true,false,true);
      }
      for(size_t i=0;i<ids.size();++i)packing.scatter(ids[i],used,values[i],result);
    }
    for(size_t i=0;i<result.size();++i)if(!should_compute_output(i))result[i]=Tensor{};
    return result;
  }
  void release_variables() override{originals.clear();leaves.clear();outputs.clear();}
  std::string name() const override{return "tide::weighted::RegionBatchVjp";}
};
std::vector<Selection> bind_group(RegionWeights w,const std::vector<RegionInput>& requests,const RegionalConfig& c,std::optional<at::Device> selection){
  RECORD_FUNCTION("tide::stage::region_bind", std::vector<c10::IValue>());
  op_profile::Scope stage_profile(op_profile::RegionBind);
  auto node=std::make_shared<RegionVjp>();variable_list inputs,leaves;
  auto leaf=[&](Tensor& t){inputs.push_back(t);t=t.detach().set_requires_grad(t.requires_grad());leaves.push_back(t);};
  std::map<const c10::TensorImpl*,Tensor> aliases;
  for(auto& [name,t]:w.extra){
    auto id=t.unsafeGetTensorImpl();auto found=aliases.find(id);
    if(found!=aliases.end())t=found->second;else{leaf(t);aliases.emplace(id,t);}
  }
  node->weights=inputs.size();node->rows=requests.size();std::vector<std::vector<Candidate>> candidates;
  for(const auto& r:requests){
    candidates.emplace_back(r.candidates.begin(),r.candidates.end());
    for(auto& d:candidates.back())for(auto& [name,t]:d.fields)inputs.push_back(t);
  }
  node->row_inputs=(inputs.size()-node->weights)/requests.size();
  node->set_next_edges(torch::autograd::collect_next_edges(inputs));
  for(const auto& t:inputs)node->originals.emplace_back(t,false);
  leaves=node->packing.pack(inputs,leaves,requests.size());
  for(const auto& t:leaves)node->leaves.emplace_back(t,false);
  at::AutoGradMode enable(true);
  // Leaves already own whole field columns. No differentiable event views
  // are created inside this adapter or passed to regional mathematics.
  std::map<std::string,std::vector<Tensor>> columns;size_t k=0;
  for(const auto& d:candidates[0])for(const auto& [name,t]:d.fields){
    auto shape=t.sizes().vec();shape.insert(shape.begin(),requests.size());
    columns[name].push_back(leaves[node->weights+k++].reshape(shape));
  }
  TensorFields fields;for(const auto& [name,values]:columns)fields[name]=at::stack(values,1);
  Tensor nodes;
  if(c.selection_mode=="regional_total_zero"||c.hard()){
    std::vector<Index> ids;for(const auto& row:candidates)for(const auto& d:row)ids.push_back(d.node);
    nodes=at::tensor(ids,fields.at("w").options().dtype(at::kLong)).reshape(fields.at("w").sizes());
  }
  RegionColumns result;
  try{result=region_columns(w,fields,c,nodes,selection);}
  catch(AllocationFailure& e){const auto& r=requests.at(e.row);e.time=r.time;e.members=r.layout.members.vec();
    for(const auto& d:r.candidates)e.nodes.push_back(d.node);throw;}
  node->mode=result.mode;node->dependencies=region_column_dependencies(result);
  stage_profile.phase(op_profile::RegionPublish);
  node->row_outputs=result.controls.size(1);
  std::vector<Tensor> saved{result.controls};
  if(result.allocation){
    const auto& a=*result.allocation;node->allocation=true;node->regularizer=a.regularizer;node->alpha=a.alpha;
    saved.insert(saved.end(),a.inputs.begin(),a.inputs.end());
    for(const auto& t:{a.controls,a.multiplier,a.branch,a.response,a.free,a.residual})saved.push_back(t);
  }
  for(const auto& t:saved)node->outputs.emplace_back(t,false);
  if(autograd_census::enabled()){
    autograd_census::graph("private/region",{result.controls});
    autograd_census::saved(inputs);autograd_census::saved(leaves);autograd_census::saved(saved);
  }
  auto support=(result.controls.detach()>0).to(at::kCPU).contiguous();auto flags=support.const_data_ptr<bool>();
  auto differentiable=region_column_differentiable(result);
  std::vector<Selection> results(requests.size());at::NoGradGuard publication_guard;
  for(size_t i=0;i<results.size();++i){
    results[i].history.last_time=requests[i].time;
    for(size_t j=0;j<node->row_outputs;++j){
      auto id=candidates[i][j].node;auto out=result.controls[i][j].detach();
      if(differentiable[i]){node->output_map.push_back(i*node->row_outputs+j);torch::autograd::set_history(out,node);}
      results[i].controls[id]=out;if(flags[i*node->row_outputs+j])results[i].active.insert(id);
    }
  }
  return results;
}
}
std::vector<Selection> region_vjp(const RegionWeights& w,const std::vector<RegionInput>& requests,const RegionalConfig& c,std::optional<at::Device> selection){
  std::map<std::string,std::vector<size_t>> groups;std::vector<Selection> result(requests.size());
  for(size_t i=0;i<requests.size();++i){
    std::ostringstream key;key<<requests[i].candidates.size();
    for(const auto& d:requests[i].candidates)for(const auto& [name,t]:d.fields)key<<'/'<<name<<t.sizes()<<t.device()<<int(t.scalar_type())<<t.requires_grad();
    groups[key.str()].push_back(i);
  }
  for(const auto& [key,ids]:groups){
    std::vector<RegionInput> rows;for(auto i:ids)rows.push_back(requests[i]);
    auto values=bind_group(w,rows,c,selection);for(size_t j=0;j<ids.size();++j)result[ids[j]]=std::move(values[j]);
  }
  return result;
}
}
