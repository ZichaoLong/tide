#pragma once
// Allocation §6/S.2. Response point and inverse bracket are separate functions.
#include "weighted_detail.h"

namespace tide::weighted::detail {
struct Regularizer {
  std::string kind;
  double alpha;
  Regularizer(std::string name,std::optional<double> shape):kind(std::move(name)),alpha(shape.value_or(0)){
    require(kind=="quad"||kind=="bern"||kind=="ent","unknown allocation regularizer");
    require(kind=="ent"?(shape&&std::isfinite(alpha)&&alpha>=1&&alpha<=2):!shape,"invalid/inapplicable ent_alpha");
  }
  Tensor response(const Tensor& x,const Tensor& mu,const Tensor& tau) const {
    auto z=x/tau;
    if(kind=="quad")return at::minimum(z.clamp_min(0),mu);
    if(kind=="bern")return mu*at::sigmoid(z);
    auto log_cap=at::where(mu>0,mu,at::ones_like(mu)).log();
    auto log_value=alpha==1?z-1:at::log1p(((alpha-1)*(z-1/alpha)).clamp_min(-1))/(alpha-1);
    auto c=at::where(log_value>=log_cap,mu,at::exp(at::minimum(log_value,log_cap)));
    return at::where(mu>0,c,at::zeros_like(c));
  }
  Tensor inverse(const Tensor& fraction,const Tensor& mu,const Tensor& tau) const {
    if(kind=="quad")return tau*fraction*mu;
    if(kind=="bern")return (tau*(fraction.log()-at::log1p(-fraction))).expand_as(mu);
    auto log_c=fraction.log()+at::where(mu>0,mu,at::ones_like(mu)).log();
    return alpha==1?tau*(1+log_c):tau*(at::expm1((alpha-1)*log_c)/(alpha-1)+1/alpha);
  }
  std::array<Tensor,3> slopes(const Tensor& x,const Tensor& mu,const Tensor& tau,const Tensor& c) const {
    Tensor a,e;
    if(kind=="bern"){
      auto p=at::sigmoid(x/tau);a=mu*p*at::sigmoid(-x/tau)/tau;e=at::where(mu>0,p,at::zeros_like(p));
    }else {
      auto free=(c>0)&(c<mu);
      a=kind=="quad"?free.to(c.scalar_type())/tau:at::where(free,at::pow(c,2-alpha)/tau,at::zeros_like(c));
      e=((c==mu)&(mu>0)).to(c.scalar_type());
    }
    return {a,e,-a*at::where(a!=0,x/tau,at::zeros_like(x))};
  }
  Tensor support(const Tensor& x,const Tensor& mu,const Tensor& tau) const {
    if(kind=="bern"||(kind=="ent"&&alpha==1))return mu>0;
    return (mu>0)&(kind=="quad"?x>0:x> -tau/(alpha*(alpha-1)));
  }
};
}
