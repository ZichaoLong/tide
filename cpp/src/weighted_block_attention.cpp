// KV §3–4: genuine shared bank roots, matrix transpose without B/H/Q/K/D.
#include "weighted_kv_block.h"
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/saved_variable.h>
#include <cmath>
#include <limits>
namespace tide::weighted::block {
namespace {
using torch::autograd::variable_list;
class Attention final:public torch::autograd::Node {
 public:
  std::vector<torch::autograd::SavedVariable> saved;
  variable_list apply(variable_list&& bars)override{
    if(at::GradMode::is_enabled())throw std::invalid_argument("KV block attention supports first-order VJP only");
    if(saved.empty())throw std::runtime_error("KV block attention graph released");
    variable_list x;for(auto& s:saved)x.push_back(s.unpack());
    variable_list result(4);if(!bars[0].defined())return result;
    const auto &q=x[0],&k=x[1],&v=x[2],&w=x[3],&p=x[4],&bar=bars[0];
    auto used=(w>0).any(1),index=used.to(at::kLong).argmax(-1).reshape({-1,1,1,1});
    auto anchor=v.gather(2,index.expand({-1,v.size(1),1,v.size(3)}));
    auto delta=at::where(used.unsqueeze(1).unsqueeze(-1),v,anchor)-anchor;
    auto mean=at::matmul(p,delta);
    auto dl=p*(at::matmul(bar,delta.transpose(-1,-2))-(bar*mean).sum(-1,true));
    const double scale=std::sqrt(double(q.size(-1)));
    if(should_compute_output(0))result[0]=at::matmul(dl,k)/scale;
    if(should_compute_output(1))result[1]=at::matmul(dl.transpose(-1,-2),q)/scale;
    if(should_compute_output(2))result[2]=at::matmul(p.transpose(-1,-2),bar);
    if(should_compute_output(3))result[3]=dl.sum(1)/at::where(w>0,w,at::ones_like(w));
    return result;
  }
  void release_variables()override{saved.clear();}
  std::string name()const override{return "tide::weighted::BlockAttentionVjp";}
};
}
Tensor attention(const Tensor& q,const Tensor& k,const Tensor& v,const Tensor& w){
  const bool grad=at::GradMode::is_enabled()&&(q.requires_grad()||k.requires_grad()||v.requires_grad()||w.requires_grad());
  at::NoGradGuard guard;auto positive=w>0,active=positive.any(-1).unsqueeze(1).unsqueeze(-1);
  auto logits=at::matmul(q,k.transpose(-1,-2))/std::sqrt(double(q.size(-1)))+
    at::where(positive,w,at::ones_like(w)).log().unsqueeze(1);
  logits=logits.masked_fill(~positive.unsqueeze(1),-std::numeric_limits<double>::infinity());
  auto p=at::softmax(at::where(active,logits,at::zeros_like(logits)),-1);
  p=at::where(active,p,at::zeros_like(p));auto result=at::matmul(p,v);
  if(grad){auto node=std::make_shared<Attention>();node->set_next_edges(torch::autograd::collect_next_edges(q,k,v,w));
    for(const auto& t:{q,k,v,w,p})node->saved.emplace_back(t,false);torch::autograd::set_history(result,node);}
  return result;
}
}
