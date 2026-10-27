// Allocation §4.2–4.4/6.3: actual complete-row batches, declared first-order VJPs.
#include "weighted_sending_batch.h"
#include "weighted_detail.h"
#include <torch/csrc/autograd/custom_function.h>

namespace tide::weighted::batch {
using detail::require;
namespace {
void validate(const Tensor& scores,const Tensor& caps){
  require(scores.dim()==2&&scores.size(0)>0&&scores.size(1)>0
    &&(scores.scalar_type()==at::kFloat||scores.scalar_type()==at::kDouble)
    &&caps.sizes()==scores.sizes()&&caps.scalar_type()==scores.scalar_type()&&caps.device()==scores.device(),"sending batch metadata mismatch");
  require(at::isfinite(scores).all().item<bool>()&&at::isfinite(caps).all().item<bool>()
    &&((caps>=0)&(caps<=1)).all().item<bool>(),"invalid batched sending scores/caps");
}
void scalar_rows(const Tensor& x,const Tensor& s){
  require(x.sizes()==at::IntArrayRef({s.size(0)})&&x.device()==s.device()&&x.scalar_type()==s.scalar_type()
    &&at::isfinite(x).all().item<bool>(),"sending batch scalar-row metadata mismatch");
}
class Shares:public torch::autograd::Function<Shares> {
 public:
  static Tensor forward(torch::autograd::AutogradContext* ctx,Tensor scores,Tensor base,Tensor tau){
    ctx->set_materialize_grads(false);auto positive=base>0,present=positive.any(1);
    auto maximum=std::get<0>(scores.masked_fill(~positive,-INFINITY).max(1));
    maximum=at::where(present,maximum,at::zeros_like(maximum));
    auto shifted=(scores-maximum.unsqueeze(1))/tau.unsqueeze(1);
    auto logs=(shifted+base.log()).masked_fill(~positive,-INFINITY);
    // Independent reduction expression with the same complete normalization.
    auto peak=std::get<0>(logs.max(1));peak=at::where(present,peak,at::zeros_like(peak));
    auto total=peak+(logs-peak.unsqueeze(1)).exp().sum(1).log();
    total=at::where(present,total,at::zeros_like(total));
    auto ratio=shifted-total.unsqueeze(1),p=at::where(positive,(ratio+base.log()).exp(),at::zeros_like(scores));
    if(!at::isfinite(p).all().item<bool>()||(positive&(p==0)).any().item<bool>())
      throw std::runtime_error("positive batched attention support underflowed in requested dtype");
    shifted=at::where(present.unsqueeze(1),shifted,at::zeros_like(shifted));
    ratio=at::where(present.unsqueeze(1),ratio,at::zeros_like(ratio));
    ctx->save_for_backward({scores,base,tau,p,shifted,ratio});return p.clone();
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,torch::autograd::variable_list bars){
    if(at::GradMode::is_enabled())throw std::invalid_argument("batched sending supports first-order VJP only");
    if(!bars[0].defined())return std::vector<Tensor>(3);
    auto s=ctx->get_saved_variables();auto base=s[1],tau=s[2],p=s[3];
    auto delta=bars[0]-(bars[0]*p).sum(1).unsqueeze(1),ds=p*delta/tau.unsqueeze(1);
    auto da=delta.sign()*(s[5]+delta.abs().log()).exp();
    da=at::where((delta!=0)&(base>0).any(1).unsqueeze(1),da,at::zeros_like(da));
    auto dt=-(ds*at::where(p>0,s[4],at::zeros_like(s[4]))).sum(1);
    for(const auto& g:{ds,da,dt})if(!at::isfinite(g).all().item<bool>())throw std::runtime_error("batched attention VJP exceeds requested dtype range");
    return {ds,da,dt};
  }
};
class Sorted:public torch::autograd::Function<Sorted> {
 public:
  static Tensor forward(torch::autograd::AutogradContext* ctx,Tensor scores,Tensor caps,Tensor target,Tensor nodes,Index k){
    ctx->set_materialize_grads(false);
    auto ids=at::argsort(nodes,true,1,false),ordered=scores.gather(1,ids).masked_fill(caps.gather(1,ids)<=0,-INFINITY);
    auto order=ids.gather(1,at::argsort(ordered,true,1,true)),cs=caps.gather(1,order),ss=scores.gather(1,order);
    auto result=at::zeros_like(caps),saturated=result.clone(),before=result.clone();
    auto partial=at::full({caps.size(0)},-1,nodes.options());
    if(k>0){
      auto positive=(cs>0)&(ss>0),chosen=positive&(positive.to(at::kLong).cumsum(1)<=k);
      auto first=(cs>0)&((cs>0).to(at::kLong).cumsum(1)==1);
      chosen=at::where(positive.any(1).unsqueeze(1),chosen,first);
      result=at::where(chosen,cs,0);saturated=chosen.to(caps.scalar_type());
    }else{
      auto remaining=target.clone();
      for(Index j=0;j<caps.size(1);++j){
        auto cj=cs.select(1,j),live=(cj>0)&(remaining>0),full=live&(remaining>=cj),part=live&~full;
        before=at::where(part.unsqueeze(1),saturated,before);partial=at::where(part,at::full_like(partial,j),partial);
        result.select(1,j).copy_(at::where(live,at::minimum(remaining,cj),0));
        saturated.select(1,j).copy_(full.to(caps.scalar_type()));remaining=at::where(live,remaining-cj,remaining);
      }
      auto complete=target==caps.sum(1);
      result=at::where(complete.unsqueeze(1),cs,result);saturated=at::where(complete.unsqueeze(1),(cs>0).to(caps.scalar_type()),saturated);
      before=at::where(complete.unsqueeze(1),at::zeros_like(before),before);partial=at::where(complete,at::full_like(partial,-1),partial);
    }
    auto inverse=at::argsort(order,true,1,false);ctx->saved_data["hard"]=k>0;
    ctx->save_for_backward({scores,caps,target,saturated,before,partial,order,inverse});return result.gather(1,inverse);
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,torch::autograd::variable_list bars){
    if(at::GradMode::is_enabled())throw std::invalid_argument("batched sending supports first-order VJP only");
    if(!bars[0].defined())return std::vector<Tensor>(5);
    auto v=ctx->get_saved_variables();auto sorted=bars[0].gather(1,v[6]);
    auto target=at::where(v[5]>=0,sorted.gather(1,v[5].clamp_min(0).unsqueeze(1)).squeeze(1),at::zeros_like(v[2]));
    auto caps=(sorted*v[3]-v[4]*target.unsqueeze(1)).gather(1,v[7]);
    return {{},caps,ctx->saved_data.at("hard").toBool()?Tensor{}:target,{},{}};
  }
};
}
Tensor attention(const Tensor& s,const Tensor& caps,const Tensor& recv,const std::string& mode,const std::string& base,const Tensor& tau){
  validate(s,caps);validate(s,recv);scalar_rows(tau,s);
  require((tau>0).all().item<bool>()&&(recv<=caps).all().item<bool>(),"invalid batched sending temperature/received");
  require((mode=="direct_attention"&&(base=="mu"||base=="received"))||(mode=="unit_attention"&&(base=="mu"||base=="uniform"))
    ||(mode=="pooled_attention"&&base=="mu"),"inapplicable batched attention binding");
  auto a=base=="mu"?caps:base=="received"?recv:at::ones_like(caps),p=Shares::apply(s,a,tau);
  auto c=mode=="direct_attention"?caps*p:mode=="pooled_attention"?caps.sum(1).clamp_max(1).unsqueeze(1)*p:p;
  if(((a>0)&(c==0)).any().item<bool>())throw std::runtime_error("positive batched attention sending support underflowed");return c;
}
Tensor proportional(const Tensor& caps,Index k){
  validate(caps,caps);require(k>0,"sending k must be positive");auto mass=caps.sum(1);
  auto scale=mass.clamp_max(k)/at::where(mass>0,mass,at::ones_like(mass));
  scale=at::where(mass>0,scale,at::ones_like(scale));auto result=caps*scale.unsqueeze(1);
  if(((caps>0)&(result==0)).any().item<bool>())throw std::runtime_error("positive proportional batch support underflowed");return result;
}
Tensor sorted_send(const Tensor& s,const Tensor& caps,const Tensor& nodes,const Tensor& target,Index k){
  validate(s,caps);scalar_rows(target,s);
  require(nodes.sizes()==s.sizes()&&nodes.device()==s.device()&&nodes.scalar_type()==at::kLong&&(nodes>=0).all().item<bool>(),"invalid batched sending node IDs");
  auto ns=std::get<0>(nodes.sort(1));require(!(ns.slice(1,1)==ns.slice(1,0,-1)).any().item<bool>(),"duplicate batched sending node IDs");
  require(k>=0&&(k>0||((target>=0)&(target<=caps.sum(1))).all().item<bool>()),"infeasible batched sending total");
  return Sorted::apply(s,caps,target,nodes,k);
}
}
