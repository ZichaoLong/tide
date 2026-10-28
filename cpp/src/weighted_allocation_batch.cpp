// Independent C++ bounded bisection across complete rows; no scalar solve replay.
#include "weighted_allocation_batch.h"
#include "tide/allocation_failure.h"
#include "weighted_quadratic_coordinates.h"
#include <ATen/core/grad_mode.h>
#include <limits>
#include <iomanip>
#include <sstream>

namespace tide::weighted::batch {
using detail::require;
namespace {
Tensor minimum(const Tensor& t){return std::get<0>(t.min(1));}
Tensor maximum(const Tensor& t){return std::get<0>(t.max(1));}
}
NumericAllocation solve(const Tensor& scores,const Tensor& caps,const Tensor& lower,const Tensor& upper,
    const Tensor& temperature,const detail::Regularizer& rule,const Tensor& shift){
  at::NoGradGuard guard;
  require(scores.dim()==2&&scores.size(0)>0&&scores.size(1)>0
    &&(scores.scalar_type()==at::kFloat||scores.scalar_type()==at::kDouble),"allocation batch requires complete FP32/FP64 rows");
  for(const auto& t:{scores,caps,lower,upper,temperature})
    require(t.device()==scores.device()&&t.scalar_type()==scores.scalar_type()&&at::isfinite(t).all().item<bool>(),"allocation batch metadata/value mismatch");
  require(caps.sizes()==scores.sizes()&&lower.sizes()==at::IntArrayRef({scores.size(0)})
    &&upper.sizes()==lower.sizes()&&temperature.sizes()==lower.sizes(),"allocation batch shape mismatch");
  require(((caps>=0)&(caps<=1)).all().item<bool>()&&(lower>=0).all().item<bool>()&&(upper>=lower).all().item<bool>()
    &&(lower<=caps.sum(1)).all().item<bool>()&&(temperature>0).all().item<bool>(),"infeasible allocation batch bounds/temperature");
  if(shift.defined())require(shift.sizes()==lower.sizes()&&shift.device()==scores.device()&&shift.scalar_type()==scores.scalar_type()
    &&at::isfinite(shift).all().item<bool>(),"invalid allocation batch shift");
  auto t=shift.defined()?scores-shift.unsqueeze(1):scores;
  require(at::isfinite(t).all().item<bool>(),"shifted batch allocation exceeds requested dtype range");
  auto mass=caps.sum(1),zero=(mass==0)|(upper==0),full=~zero&(lower==mass),normal=~zero&~full;
  NumericAllocation out{at::zeros_like(t),at::zeros_like(lower),at::full_like(lower,Slack,at::kLong),t.clone(),{},at::zeros_like(lower)};
  out.branch.index_put_({zero},Zero);out.branch.index_put_({full},Caps);out.controls.index_put_({full},caps.index({full}));
  auto ids=at::nonzero(normal).flatten();
  if(ids.numel()){
    auto tn=t.index({ids}),cn=caps.index({ids}),tau=temperature.index({ids}).unsqueeze(1);
    auto controls=rule.response(tn,cn,tau),sum=controls.sum(1);
    auto lo=lower.index({ids}),hi=upper.index({ids});
    auto branch=at::where(lo==hi,Equality,at::where(sum<lo,Lower,at::where(sum>hi,Upper,Slack)));
    out.branch.index_put_({ids},branch);out.controls.index_put_({ids},controls);
    auto rows=ids.index({branch!=Slack});
    if(rows.numel()){
      auto target=at::where((out.branch.index({rows})==Lower)|(out.branch.index({rows})==Equality),lower.index({rows}),upper.index({rows}));
      auto ts=scores.index({rows}),mu=caps.index({rows}),tt=temperature.index({rows}).unsqueeze(1);
      auto center=rule.kind=="quad"?detail::quad_pivot(ts,mu,target):maximum(ts),centered=ts-center.unsqueeze(1);
      Tensor a,b;
      if(rule.kind=="quad"){
        a=-tt.squeeze(1)*maximum(mu);b=at::zeros_like(a);
        if(!(a<0).all().item<bool>())throw std::runtime_error("quadratic transition interval underflowed in requested dtype");
      }
      else{
        auto points=centered-rule.inverse((target/mass.index({rows})).unsqueeze(1),mu,tt),positive=mu>0;
        a=minimum(points.masked_fill(~positive,INFINITY));b=maximum(points.masked_fill(~positive,-INFINITY));
      }
      for(int iteration=0;iteration<80;++iteration){
        auto middle=a+(b-a)/2;
        if(iteration%8==0&&((middle==a)|(middle==b)).all().item<bool>())break;
        auto above=rule.response(centered-middle.unsqueeze(1),mu,tt).sum(1)>target;
        a=at::where(above,middle,a);b=at::where(above,b,middle);
      }
      auto lam=a+(b-a)/2;
      if(rule.kind=="quad"){
        auto c=rule.response(centered-lam.unsqueeze(1),mu,tt),free=(c>0)&(c<mu),count=free.sum(1);
        auto capped=(c==mu)&~free;
        auto exact=(at::where(free,centered,0).sum(1)+tt.squeeze(1)*(at::where(capped,mu,0).sum(1)-target))/count.clamp_min(1);
        lam=at::where(count>0,exact,lam);
      }else{
        std::vector<Tensor> values{a,lam,b},errors;
        for(const auto& v:values)errors.push_back((rule.response(centered-v.unsqueeze(1),mu,tt).sum(1)-target).abs());
        auto chosen=at::stack(errors,1).argmin(1);
        lam=at::stack(values,1).gather(1,chosen.unsqueeze(1)).squeeze(1);
      }
      auto x=centered-lam.unsqueeze(1);
      out.controls.index_put_({rows},rule.response(x,mu,tt));
      auto multiplier=lam+center;if(shift.defined())multiplier=multiplier-shift.index({rows});
      out.multiplier.index_put_({rows},multiplier);
      out.response.index_put_({rows},x);
    }
    auto c=out.controls.index({ids});
    auto x=out.response.index({ids});
    auto error=maximum((c-rule.response(x,cn,tau)).abs());
    auto residual=maximum(at::stack({error,(lo-c.sum(1)).clamp_min(0),(c.sum(1)-hi).clamp_min(0)},1));
    out.residual.index_put_({ids},residual);
    const double eps=scores.scalar_type()==at::kDouble?std::numeric_limits<double>::epsilon():std::numeric_limits<float>::epsilon();
    auto reject=[&](const Tensor& mask,const std::string& message){
      if(!mask.any().item<bool>())return;auto bad=at::nonzero(mask).flatten();
      const auto row=ids[bad[0]].item<Index>();
      std::ostringstream shape;shape<<std::setprecision(std::numeric_limits<double>::max_digits10)<<rule.alpha;
      throw AllocationFailure(message,row,{{"scores",scores[row]},{"caps",caps[row]},
        {"lower",lower[row]},{"upper",upper[row]},{"temperature",temperature[row]},
        {"shift",shift.defined()?shift[row]:Tensor{}},{"controls",out.controls[row]},
        {"response",out.response[row]},{"multiplier",out.multiplier[row]},
        {"residual",out.residual[row]},{"tolerance",at::full({},64*eps*scores.size(1),scores.options())}},
        {{"regularizer",rule.kind},{"alpha",shape.str()},
         {"branch",branches[out.branch[row].item<Index>()]},{"solver","native-batch"}});
    };
    reject(~at::isfinite(residual)|(residual>64*eps*scores.size(1)),"allocation batch residual exceeds dtype tolerance");
    reject((rule.support(x,cn,tau)&(c==0)).any(1),"positive allocation batch support underflowed in requested dtype");
  }
  out.free=(out.controls>0)&(out.controls<caps);return out;
}
}
