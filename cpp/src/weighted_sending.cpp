// Independent C++ sending formulas; no Python or legacy Emit/HST dependency.
#include "tide/weighted_sending.h"
#include "weighted_detail.h"
#include <torch/csrc/autograd/custom_function.h>
#include <algorithm>
#include <set>

namespace tide::weighted {
using namespace detail;
namespace {
void vectors(const Tensor& s,const Tensor& a,bool caps=false){
  require(s.defined()&&s.dim()==1&&(s.scalar_type()==at::kFloat||s.scalar_type()==at::kDouble)
    &&a.defined()&&a.sizes()==s.sizes()&&a.device()==s.device()&&a.scalar_type()==s.scalar_type(),
    "sending requires matching FP32/FP64 vectors");
  require(at::isfinite(s).all().item<bool>()&&at::isfinite(a).all().item<bool>()
    &&(a>=0).all().item<bool>()&&(!caps||(a<=1).all().item<bool>()),"invalid sending scores/weights/caps");
}
void scalar(const Tensor& x,const Tensor& s){
  require(x.defined()&&x.dim()==0&&x.scalar_type()==s.scalar_type()&&x.device()==s.device()
    &&at::isfinite(x).item<bool>(),"sending scalar dtype/device/value mismatch");
}
class Shares:public torch::autograd::Function<Shares> {
 public:
  static Tensor forward(torch::autograd::AutogradContext* ctx,Tensor s,Tensor a,Tensor tau){
    ctx->set_materialize_grads(false);auto positive=a>0,p=at::zeros_like(s),z=p.clone(),log_ratio=p.clone();
    if(positive.any().item<bool>()){
      z=(s-s.index({positive}).max())/tau;
      auto logs=z.index({positive})+a.index({positive}).log();
      auto shift=logs.max(),log_total=shift+(logs-shift).exp().sum().log();
      log_ratio=z-log_total;p.index_put_({positive},(log_ratio.index({positive})+a.index({positive}).log()).exp());
      if(!at::isfinite(p).all().item<bool>()||(p.index({positive})==0).any().item<bool>())
        throw std::runtime_error("positive attention support underflowed in requested dtype");
    }
    ctx->save_for_backward({s,a,tau,p,z,log_ratio});return p.clone();
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,torch::autograd::variable_list gs){
    if(!gs[0].defined())return std::vector<Tensor>(3);
    at::AutoGradMode first_order(false);auto v=ctx->get_saved_variables();auto s=v[0],a=v[1],tau=v[2],p=v[3];
    if(!(a>0).any().item<bool>())return {at::zeros_like(s),at::zeros_like(a),at::zeros_like(tau)};
    auto d=gs[0]-(p*gs[0]).sum(),ds=p*d/tau;
    auto da=at::where(d!=0,d.sign()*(v[5]+d.abs().log()).exp(),at::zeros_like(d));
    auto dt=-(ds*at::where(p>0,v[4],at::zeros_like(v[4]))).sum();
    for(const auto& g:{ds,da,dt})if(!at::isfinite(g).all().item<bool>())throw std::runtime_error("attention VJP exceeds requested dtype range");
    return {ds,da,dt};
  }
};
std::vector<Index> order(const Tensor& s,const Tensor& mu,const std::vector<Index>& nodes){
  require(static_cast<Index>(nodes.size())==s.numel()&&std::set<Index>(nodes.begin(),nodes.end()).size()==nodes.size(),
    "sending node IDs must be distinct nonnegative int64");
  std::vector<Index> indices;std::vector<double> scores;
  for(Index i=0;i<s.numel();++i){
    require(nodes[i]>=0,"sending node IDs must be nonnegative");scores.push_back(s[i].item<double>());
    if(mu[i].item<double>()>0)indices.push_back(i);
  }
  std::sort(indices.begin(),indices.end(),[&](Index i,Index j){return scores[i]!=scores[j]?scores[i]>scores[j]:nodes[i]<nodes[j];});
  return indices;
}
class SortedSending:public torch::autograd::Function<SortedSending> {
 public:
  static Tensor forward(torch::autograd::AutogradContext* ctx,Tensor s,Tensor mu,Tensor total,std::vector<Index> ordered,Index k){
    ctx->set_materialize_grads(false);auto c=at::zeros_like(mu),sat=c.clone(),before=c.clone();Index partial=-1;
    if(k>0){
      Index used=0;
      for(auto i:ordered)if(s[i].item<double>()>0&&used<k){c[i]=mu[i];sat[i]=1;++used;}
      if(used==0&&!ordered.empty()){auto i=ordered[0];c[i]=mu[i];sat[i]=1;}
    } else if(total.item<double>()==mu.sum().item<double>()){
      c=mu.clone();sat=(mu>0).to(mu.scalar_type());
    } else {
      auto remaining=total.clone();
      for(auto i:ordered){
        if(remaining.item<double>()<=0)break;
        if(remaining.item<double>()>=mu[i].item<double>()){c[i]=mu[i];sat[i]=1;}
        else {c[i]=remaining;partial=i;before=sat.clone();}
        remaining=remaining-mu[i];
      }
    }
    ctx->save_for_backward({s,mu,total,sat,before});ctx->saved_data["partial"]=partial;ctx->saved_data["hard"]=k>0;return c;
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,torch::autograd::variable_list gs){
    if(!gs[0].defined())return std::vector<Tensor>(5);
    at::AutoGradMode first_order(false);auto v=ctx->get_saved_variables();auto i=ctx->saved_data.at("partial").toInt();
    auto gt=i>=0?gs[0][i]:at::zeros_like(v[2]);
    return {{},gs[0]*v[3]-v[4]*gt,ctx->saved_data.at("hard").toBool()?Tensor{}:gt,{},{}};
  }
};
}
Tensor attention_shares(const Tensor& s,const Tensor& a,const Tensor& tau){
  vectors(s,a);scalar(tau,s);require(tau.item<double>()>0,"attention temperature must be positive");
  return Shares::apply(s,a,tau);
}
Tensor attention_send(const Tensor& s,const Tensor& mu,const Tensor& w,const std::string& mode,const std::string& base,const Tensor& tau){
  vectors(s,mu,true);vectors(s,w,true);require((w<=mu).all().item<bool>(),"received exceeds sending caps");
  require((mode=="direct_attention"&&(base=="mu"||base=="received"))||
    (mode=="unit_attention"&&(base=="mu"||base=="uniform"))||(mode=="pooled_attention"&&base=="mu"),
    "inapplicable attention mode/allocation_base");
  auto a=base=="mu"?mu:base=="received"?w:at::ones_like(mu),p=attention_shares(s,a,tau);
  auto c=mode=="direct_attention"?mu*p:mode=="pooled_attention"?mu.sum().clamp_max(1)*p:p;
  if(((a>0)&(c==0)).any().item<bool>())throw std::runtime_error("positive attention sending support underflowed in requested dtype");
  return c;
}
Tensor proportional(const Tensor& mu,Index k){
  vectors(mu,mu,true);require(k>0,"sending k must be positive");auto mass=mu.sum();
  if(mass.item<double>()==0)return mu;
  auto c=mu*(mass.clamp_max(k)/mass);
  if(((mu>0)&(c==0)).any().item<bool>())throw std::runtime_error("positive proportional support underflowed in requested dtype");
  return c;
}
Tensor total_zero(const Tensor& s,const Tensor& mu,const Tensor& total,const std::vector<Index>& nodes){
  vectors(s,mu,true);scalar(total,s);
  require(total.item<double>()>=0&&total.item<double>()<=mu.sum().item<double>(),"infeasible zero-temperature sending total");
  return SortedSending::apply(s,mu,total,order(s,mu,nodes),Index{0});
}
Tensor hard_selection(const Tensor& s,const Tensor& mu,Index k,const std::vector<Index>& nodes){
  vectors(s,mu,true);require(k>0,"sending k must be positive");
  return SortedSending::apply(s,mu,at::zeros({},s.options()),order(s,mu,nodes),k);
}
}
