#pragma once
// Architecture §2–4: no cross-row connectivity exists in a singleton batch.
// Preserve first-order/None roots and original version checks without private
// detached leaves or a nested autograd::grad invocation.
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/saved_variable.h>

namespace tide::weighted {
class SingletonVjp final:public torch::autograd::Node {
 public:
  size_t count=0,total=0;
  std::vector<size_t> output_map;
  std::vector<torch::autograd::SavedVariable> originals;
  torch::autograd::variable_list apply(torch::autograd::variable_list&& bars) override {
    if(at::GradMode::is_enabled())throw std::invalid_argument("weighted singleton supports first-order VJP only");
    if(originals.empty())throw std::runtime_error("singleton VJP graph already released; use retain_graph");
    for(auto& t:originals)t.unpack();
    torch::autograd::variable_list result(total);
    for(size_t i=0;i<bars.size();++i)result[count+output_map[i]]=bars[i];
    return result;
  }
  void release_variables() override {originals.clear();}
  std::string name() const override {return "tide::weighted::SingletonVjp";}
};
inline std::vector<Tensor> singleton_boundary(const std::vector<Tensor>& originals,const std::vector<Tensor>& refs){
  auto node=std::make_shared<SingletonVjp>();
  auto inputs=originals;inputs.insert(inputs.end(),refs.begin(),refs.end());
  node->count=originals.size();node->total=inputs.size();
  node->set_next_edges(torch::autograd::collect_next_edges(inputs));
  for(const auto& t:originals)node->originals.emplace_back(t,false);
  std::vector<Tensor> result;
  for(size_t i=0;i<refs.size();++i){
    auto value=refs[i].detach().clone();
    if(refs[i].requires_grad()){node->output_map.push_back(i);torch::autograd::set_history(value,node);}
    result.push_back(value);
  }
  return result;
}
}
