#include "weighted_row_linear.h"
#include "tide/packing_work.h"
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/saved_variable.h>
#include <stdexcept>

namespace tide::weighted {
namespace {
using torch::autograd::variable_list;
class LinearVjp final:public torch::autograd::Node {
 public:
  std::vector<torch::autograd::SavedVariable> originals,numeric;
  std::vector<size_t> output_map;
  variable_list apply(variable_list&& bars) override {
    if(at::GradMode::is_enabled())throw std::invalid_argument("independent linear rows support first-order VJP only");
    if(originals.empty())throw std::runtime_error("linear row graph already released; use retain_graph");
    for(auto& v:originals)v.unpack(); // Original parameter/input version checks.
    variable_list result(originals.size()),gs;std::vector<Index> ids;
    for(size_t i=0;i<bars.size();++i)if(bars[i].defined()){ids.push_back(output_map[i]);gs.push_back(bars[i]);}
    if(!ids.empty()){
      auto x=numeric[0].unpack(),w=numeric[1].unpack(),index=at::tensor(ids,x.options().dtype(at::kLong));
      auto g=at::stack(gs),dx=at::bmm(w.index_select(0,index).transpose(1,2),g.unsqueeze(-1)).squeeze(-1);
      auto dw=g.unsqueeze(-1)*x.index_select(0,index).unsqueeze(1);
      for(size_t k=0;k<ids.size();++k){result[2*ids[k]]=dx[k];result[2*ids[k]+1]=dw[k];}
    }
    for(size_t i=0;i<result.size();++i)if(!should_compute_output(i))result[i]=Tensor{};
    return result;
  }
  void release_variables() override{originals.clear();numeric.clear();}
  std::string name() const override{return "tide::weighted::IndependentLinearRowsVjp";}
};
}
std::vector<Tensor> linear_rows(const std::vector<Tensor>& xs,const std::vector<Tensor>& weights){
  if(xs.size()!=weights.size())throw std::invalid_argument("independent linear row count mismatch");
  if(xs.empty())return {};
  const auto& x0=xs[0];const auto& w0=weights[0];
  if(x0.dim()!=1||w0.dim()!=2||w0.size(1)!=x0.numel())throw std::invalid_argument("independent linear row shape mismatch");
  variable_list inputs;
  for(size_t i=0;i<xs.size();++i){
    const auto& x=xs[i];const auto& w=weights[i];
    if(x.sizes()!=x0.sizes()||w.sizes()!=w0.sizes()||x.device()!=x0.device()||w.device()!=x0.device()
       ||x.scalar_type()!=x0.scalar_type()||w.scalar_type()!=x0.scalar_type())
      throw std::invalid_argument("independent linear row metadata mismatch");
    inputs.push_back(x);inputs.push_back(w);
  }
  Tensor x,w,values;
  work::packing::projection("private_bmm",xs.size());
  {at::NoGradGuard guard;x=at::stack(xs);w=at::stack(weights);values=at::bmm(w,x.unsqueeze(-1)).squeeze(-1);}
  auto node=std::make_shared<LinearVjp>();node->set_next_edges(torch::autograd::collect_next_edges(inputs));
  for(const auto& t:inputs)node->originals.emplace_back(t,false);
  for(const auto& t:{x,w})node->numeric.emplace_back(t,false);
  std::vector<Tensor> outputs;
  for(size_t i=0;i<xs.size();++i){
    auto out=values[i];
    if(at::GradMode::is_enabled()&&(xs[i].requires_grad()||weights[i].requires_grad())){
      node->output_map.push_back(i);torch::autograd::set_history(out,node);
    }
    outputs.push_back(out);
  }
  return outputs;
}
}
