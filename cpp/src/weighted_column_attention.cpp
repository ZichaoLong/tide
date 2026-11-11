// KV positional logits; one shared value bank and its analytical transpose.
#include "tide/weighted_sequence_columns.h"
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/saved_variable.h>
#include <limits>
namespace tide::weighted::columns {
namespace {
using torch::autograd::variable_list;
class LogitAttention final:public torch::autograd::Node {
 public:
  std::vector<torch::autograd::SavedVariable> saved;
  variable_list apply(variable_list&& bars)override{
    if(at::GradMode::is_enabled())throw std::invalid_argument("KV positional attention supports first-order VJP only");
    if(saved.empty())throw std::runtime_error("KV positional attention graph released");
    variable_list x;for(auto& s:saved)x.push_back(s.unpack());
    variable_list result(3);if(!bars[0].defined())return result;
    const auto &v=x[1],&w=x[2],&p=x[3],&bar=bars[0];
    auto used=(w>0).any(1),index=used.to(at::kLong).argmax(-1).reshape({-1,1,1,1});
    auto anchor=v.gather(2,index.expand({-1,v.size(1),1,v.size(3)}));
    auto delta=at::where(used.unsqueeze(1).unsqueeze(-1),v,anchor)-anchor;
    auto dl=p*(at::matmul(bar,delta.transpose(-1,-2))-(bar*at::matmul(p,delta)).sum(-1,true));
    dl=at::where((w>0).sum(-1).unsqueeze(1).unsqueeze(-1)>1,dl,at::zeros_like(dl));
    if(should_compute_output(0))result[0]=dl;
    if(should_compute_output(1))result[1]=at::matmul(p.transpose(-1,-2),bar);
    if(should_compute_output(2))result[2]=dl.sum(1)/at::where(w>0,w,at::ones_like(w));
    return result;
  }
  void release_variables()override{saved.clear();}
  std::string name()const override{return "tide::weighted::ColumnLogitAttentionVjp";}
};
}
Tensor logit_attention(const Tensor& logits,const Tensor& v,const Tensor& w){
  const bool grad=at::GradMode::is_enabled()&&(logits.requires_grad()||v.requires_grad()||w.requires_grad());
  at::NoGradGuard guard;auto positive=w>0,active=positive.any(-1).unsqueeze(1).unsqueeze(-1);
  auto l=logits+at::where(positive,w,at::ones_like(w)).log().unsqueeze(1);
  l=l.masked_fill(~positive.unsqueeze(1),-std::numeric_limits<double>::infinity());
  auto p=at::softmax(at::where(active,l,at::zeros_like(l)),-1);
  p=at::where(active,p,at::zeros_like(p));auto result=at::matmul(p,v);
  if(grad){auto node=std::make_shared<LogitAttention>();node->set_next_edges(torch::autograd::collect_next_edges(logits,v,w));
    for(const auto& t:{logits,v,w,p})node->saved.emplace_back(t,false);torch::autograd::set_history(result,node);}
  return result;
}
}
