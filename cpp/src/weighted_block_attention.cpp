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
    auto q=x[0],k=x[1],v=x[2],w=x[3],p=x[4],bar=bars[0];
    if(q.scalar_type()==at::kHalf){q=q.to(at::kFloat);k=k.to(at::kFloat);
      v=v.to(at::kFloat);w=w.to(at::kFloat);bar=bar.to(at::kFloat);}
    auto used=(w>0).any(1),index=used.to(at::kLong).argmax(-1).reshape({-1,1,1,1});
    auto anchor=v.gather(2,index.expand({-1,v.size(1),1,v.size(3)}));
    auto delta=at::where(used.unsqueeze(1).unsqueeze(-1),v,anchor)-anchor;
    auto mean=at::matmul(p,delta);
    auto dl=p*(at::matmul(bar,delta.transpose(-1,-2))-(bar*mean).sum(-1,true));
    dl=at::where((w>0).sum(-1).unsqueeze(1).unsqueeze(-1)>1,dl,at::zeros_like(dl));
    const double scale=std::sqrt(double(q.size(-1)));
    if(should_compute_output(0))result[0]=at::matmul(dl,k)/scale;
    if(should_compute_output(1))result[1]=at::matmul(dl.transpose(-1,-2),q)/scale;
    if(should_compute_output(2))result[2]=at::matmul(p.transpose(-1,-2),bar);
    if(should_compute_output(3))result[3]=dl.sum(1)/at::where(w>0,w,at::ones_like(w));
    for(size_t j=0;j<result.size();++j)if(result[j].defined())result[j]=result[j].to(x[j].scalar_type());
    return result;
  }
  void release_variables()override{saved.clear();}
  std::string name()const override{return "tide::weighted::BlockAttentionVjp";}
};
}
Tensor attention(const Tensor& original_q,const Tensor& original_k,const Tensor& original_v,const Tensor& original_w){
  const auto& oq=original_q;const auto& ok=original_k;const auto& ov=original_v;const auto& ow=original_w;
  auto q=oq,k=ok,v=ov,w=ow;
  const bool grad=at::GradMode::is_enabled()&&(q.requires_grad()||k.requires_grad()||v.requires_grad()||w.requires_grad());
  at::NoGradGuard guard;
  if(q.scalar_type()==at::kHalf){q=q.to(at::kFloat);k=k.to(at::kFloat);v=v.to(at::kFloat);w=w.to(at::kFloat);}
  auto positive=w>0,active=positive.any(-1).unsqueeze(1).unsqueeze(-1);
  auto logits=at::matmul(q,k.transpose(-1,-2))/std::sqrt(double(q.size(-1)))+
    at::where(positive,w,at::ones_like(w)).log().unsqueeze(1);
  logits=logits.masked_fill(~positive.unsqueeze(1),-std::numeric_limits<double>::infinity());
  auto p=at::softmax(at::where(active,logits,at::zeros_like(logits)),-1);
  p=at::where(active,p,at::zeros_like(p));auto result=at::matmul(p,v).to(ov.scalar_type());
  if(grad){auto node=std::make_shared<Attention>();node->set_next_edges(torch::autograd::collect_next_edges(oq,ok,ov,ow));
    for(const auto& t:{oq,ok,ov,ow,p})node->saved.emplace_back(t,false);torch::autograd::set_history(result,node);}
  return result;
}
}
