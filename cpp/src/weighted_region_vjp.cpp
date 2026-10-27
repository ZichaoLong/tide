// Allocation §2–6: independent frame/control roots on actual batch graphs.
#include "weighted_region_batch.h"
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
  std::vector<torch::autograd::SavedVariable> originals,leaves,outputs;
  std::vector<std::string> dependencies;
  std::vector<size_t> output_map;
  variable_list apply(variable_list&& bars) override {
    if(at::GradMode::is_enabled())throw std::invalid_argument("batched region supports first-order VJP only");
    if(originals.empty())throw std::runtime_error("regional VJP graph already released; use retain_graph");
    for(auto& t:originals)t.unpack();variable_list xs,ys,full(rows*row_outputs);
    for(auto& t:leaves)xs.push_back(t.unpack());for(auto& t:outputs)ys.push_back(t.unpack());
    for(size_t i=0;i<bars.size();++i)full[output_map[i]]=bars[i];
    using Key=std::pair<std::vector<size_t>,std::string>;std::map<Key,std::vector<size_t>> groups;
    for(size_t row=0;row<rows;++row){
      std::vector<size_t> mask;for(size_t j=0;j<row_outputs;++j)if(full[row*row_outputs+j].defined())mask.push_back(j);
      if(!mask.empty())groups[{mask,dependencies[row]}].push_back(row);
    }
    variable_list result(xs.size());
    for(const auto& [key,used]:groups){
      variable_list roots,grads,inputs;std::vector<size_t> ids;
      for(auto row:used)for(auto j:key.first){roots.push_back(ys[row*row_outputs+j]);grads.push_back(full[row*row_outputs+j]);}
      auto append=[&](size_t i){if(xs[i].requires_grad()&&should_compute_output(i)){ids.push_back(i);inputs.push_back(xs[i]);}};
      for(size_t i=0;i<weights;++i)append(i);for(auto row:used)for(size_t j=0;j<row_inputs;++j)append(weights+row*row_inputs+j);
      if(inputs.empty())continue;
      auto values=torch::autograd::grad(roots,inputs,grads,true,false,true);
      for(size_t i=0;i<ids.size();++i)if(values[i].defined()){
        auto& out=result[ids[i]];out=out.defined()?out+values[i]:values[i];
      }
    }
    return result;
  }
  void release_variables() override{originals.clear();leaves.clear();outputs.clear();}
  std::string name() const override{return "tide::weighted::RegionBatchVjp";}
};
std::vector<Selection> bind_group(RegionWeights w,const std::vector<RegionInput>& requests,const RegionalConfig& c,std::optional<at::Device> selection){
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
    for(auto& d:candidates.back())for(auto& [name,t]:d.fields)leaf(t);
  }
  node->row_inputs=(inputs.size()-node->weights)/requests.size();
  node->set_next_edges(torch::autograd::collect_next_edges(inputs));
  for(const auto& t:inputs)node->originals.emplace_back(t,false);for(const auto& t:leaves)node->leaves.emplace_back(t,false);
  std::vector<RegionInput> bound;
  for(size_t i=0;i<requests.size();++i){const auto& r=requests[i];bound.push_back({r.history,r.time,candidates[i],r.layout,r.payload_options});}
  std::vector<Selection> results;
  {at::AutoGradMode enable(true);results=regions(w,bound,c,&node->dependencies,selection);}
  node->row_outputs=results[0].controls.size();
  for(size_t i=0;i<results.size();++i){
    size_t j=0;for(auto& [id,t]:results[i].controls){
      node->outputs.emplace_back(t,false);auto out=t.detach().clone();
      if(t.requires_grad()){node->output_map.push_back(i*node->row_outputs+j);torch::autograd::set_history(out,node);}
      t=out;++j;
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
