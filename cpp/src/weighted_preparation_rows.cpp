#include "weighted_preparation_rows.h"
#include "weighted_packed.h"
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/saved_variable.h>
#include <set>

namespace tide::weighted {
namespace {
using torch::autograd::variable_list;
class PreparationVjp final:public torch::autograd::Node {
 public:
  size_t parameters=0,width=0;
  std::vector<torch::autograd::SavedVariable> originals,leaves,outputs;
  std::vector<size_t> output_map;
  variable_list apply(variable_list&& bars) override {
    if(at::GradMode::is_enabled())throw std::invalid_argument("KV preparation supports first-order VJP only");
    if(originals.empty())throw std::runtime_error("KV preparation graph released; use retain_graph");
    for(auto& v:originals)v.unpack();
    std::set<size_t> rows;
    variable_list roots,gradients,inputs,result(originals.size());
    for(size_t i=0;i<bars.size();++i)if(bars[i].defined()){
      auto j=output_map[i];rows.insert(j/2);roots.push_back(outputs[j].unpack());gradients.push_back(bars[i]);
    }
    std::vector<size_t> ids;
    auto append=[&](size_t i){auto t=leaves[i].unpack();if(t.requires_grad()&&should_compute_output(i)){
      ids.push_back(i);inputs.push_back(t);}};
    for(size_t i=0;i<parameters;++i)append(i);
    for(auto r:rows)for(size_t j=0;j<width;++j)append(parameters+r*width+j);
    if(!roots.empty()&&!inputs.empty()){
      auto grads=torch::autograd::grad(roots,inputs,gradients,true,false,true);
      for(size_t i=0;i<ids.size();++i)result[ids[i]]=grads[i];
    }
    return result;
  }
  void release_variables() override{originals.clear();leaves.clear();outputs.clear();}
  std::string name() const override{return "tide::weighted::KVPreparationRowsVjp";}
};
}
std::vector<std::pair<Tensor,Tensor>> prepare_kv_rows(bool common,const std::vector<Tensor>& weights,
    const std::vector<std::pair<Tensor,Tensor>>& rows,double epsilon){
  if(rows.empty())return {};
  variable_list inputs=weights,leaves;
  for(const auto& row:rows){inputs.push_back(row.first);if(!common)inputs.push_back(row.second);}
  for(const auto& t:inputs)leaves.push_back(t.detach().set_requires_grad(t.requires_grad()));
  const size_t count=weights.size(),width=common?1:2;
  Tensor a,b;
  {
    at::AutoGradMode enable(true);
    variable_list xs,bs;
    for(size_t i=0;i<rows.size();++i){xs.push_back(leaves[count+i*width]);if(!common)bs.push_back(leaves[count+i*width+1]);}
    if(common){a=normalize(at::stack(xs).slice(1,1),epsilon);b=packed::project(a,leaves[0]);}
    else {auto x=at::stack(xs);a=at::stack(bs)+packed::project(packed::project(x,leaves[0]),leaves[1]);
      b=packed::project(x,leaves[2]);}
  }
  auto node=std::make_shared<PreparationVjp>();node->parameters=count;node->width=width;
  node->set_next_edges(torch::autograd::collect_next_edges(inputs));
  for(const auto& t:inputs)node->originals.emplace_back(t,false);
  for(const auto& t:leaves)node->leaves.emplace_back(t,false);
  std::vector<std::pair<Tensor,Tensor>> result;
  // Create the saved row views with grad enabled; published values are detached.
  variable_list as,bs;
  {at::AutoGradMode enable(true);as=a.unbind();bs=b.unbind();}
  for(size_t i=0;i<rows.size();++i){
    std::pair<Tensor,Tensor> row;
    for(size_t j=0;j<2;++j){
      auto root=j?bs[i]:as[i];node->outputs.emplace_back(root,false);auto out=root.detach();
      if(at::GradMode::is_enabled()&&root.requires_grad()){
        node->output_map.push_back(2*i+j);torch::autograd::set_history(out,node);
      }
      (j?row.second:row.first)=out;
    }
    result.push_back(std::move(row));
  }
  return result;
}
}
