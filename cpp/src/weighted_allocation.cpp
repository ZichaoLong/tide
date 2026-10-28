// Allocation §6/S.2: independent bracket solver + exact response-map VJP.
#include "tide/weighted.h"
#include <torch/csrc/autograd/custom_function.h>
#include <limits>
#include <stdexcept>

namespace tide::weighted {
namespace {
Tensor response(const Tensor& t,const Tensor& mu,const Tensor& tau,const Tensor& lam) {
  return at::minimum(((t-lam)/tau).clamp_min(0),mu);
}
class ExactQuad:public torch::autograd::Function<ExactQuad> {
 public:
  static Tensor forward(torch::autograd::AutogradContext* ctx,Tensor t,Tensor mu,Tensor target,
                        Tensor tau,Tensor c,Tensor lam,bool equality,Tensor shift,Tensor response) {
    ctx->set_materialize_grads(false);ctx->save_for_backward({t,mu,target,tau,c,lam,shift,response});
    ctx->saved_data["equality"]=equality;return c.clone();
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                torch::autograd::variable_list grads) {
    if(!grads[0].defined()) return std::vector<Tensor>(9);
    at::AutoGradMode first_order(false);
    auto s=ctx->get_saved_variables();auto mu=s[1],tau=s[3],c=s[4],shift=s[6],response=s[7];
    auto a=((c>0)&(c<mu)).to(c.scalar_type())/tau;
    auto e=((c==mu)&(mu>0)).to(c.scalar_type()), total=a.sum();
    const bool equality=ctx->saved_data.at("equality").toBool();auto mean=at::zeros({},c.options()),centered=grads[0];
    if(equality&&total.item<double>()>0){auto anchor=grads[0][a.argmax().item<Index>()],relative=grads[0]-anchor,offset=(a*relative).sum()/total;mean=anchor+offset;centered=relative-offset;}
    auto gt=a*centered;
    auto gs=shift.defined()?(equality?at::zeros({},c.options()):-gt.sum()):Tensor{};
    return {gt,e*centered,mean,(-a*at::where(a!=0,response/tau,0)*centered).sum(),{},{},{},gs,{}};
  }
};
void check(bool ok,const char* text){if(!ok)throw std::invalid_argument(text);}
}
Allocation allocate_quad(const Tensor& input,const Tensor& mu,const Tensor& lower,
                         const Tensor& upper,const Tensor& tau,const Tensor& shift) {
  auto t=input;
  check(t.defined()&&t.dim()==1&&t.numel()>0&&(t.scalar_type()==at::kFloat||t.scalar_type()==at::kDouble),
        "weighted allocation requires FP32/FP64 scores");
  for(const auto& x:{mu,lower,upper,tau})
    check(x.defined()&&x.scalar_type()==t.scalar_type()&&x.device()==t.device()&&at::isfinite(x).all().item<bool>(),
          "weighted allocation dtype/device/nonfinite input");
  check(mu.sizes()==t.sizes()&&lower.dim()==0&&upper.dim()==0&&tau.dim()==0&&at::isfinite(t).all().item<bool>(),
        "weighted allocation shape/nonfinite scores");
  check((mu>=0).all().item<bool>()&&(mu<=1).all().item<bool>()&&lower.item<double>()>=0
        &&upper.item<double>()>=lower.item<double>()&&lower.item<double>()<=mu.sum().item<double>()&&tau.item<double>()>0,
        "infeasible weighted allocation bounds/temperature");
  if(shift.defined())check(shift.dim()==0&&shift.device()==t.device()&&shift.scalar_type()==t.scalar_type()
    &&at::isfinite(shift).item<bool>(),"invalid allocation score_shift");
  Allocation result;Tensor response_input;
  {
    at::NoGradGuard guard;
    if(shift.defined()){t=input-shift;check(at::isfinite(t).all().item<bool>(),"shifted allocation scores exceed requested dtype range");}
    response_input=t;
    auto mass=mu.sum(),zero=at::zeros({},t.options());result.multiplier=zero;
    if(mass.item<double>()==0||upper.item<double>()==0){result.controls=at::zeros_like(t);result.constraint="zero";}
    else if(lower.item<double>()==mass.item<double>()){result.controls=mu.clone();result.constraint="caps";}
    else {
      auto c=response(t,mu,tau,zero);auto sum=c.sum().item<double>();
      auto branch=sum<lower.item<double>()?"lower":sum>upper.item<double>()?"upper":"slack";
      if(lower.item<double>()==upper.item<double>())branch="equality";
      result.constraint=branch;
      if(result.constraint!="slack") {
        auto target=(result.constraint=="lower"||result.constraint=="equality")?lower:upper;
        auto shift=t.max(), centered=t-shift;
        auto point=centered-tau*(target/mass)*mu;
        auto positive=mu>0;
        auto lo=point.index({positive}).min(),hi=point.index({positive}).max();
        // Fixed bounded bisection is forward numerics only. The backward below
        // differentiates the exact map, independently of these iterations.
        for(int i=0;i<80;++i){
          auto middle=lo+(hi-lo)/2;
          auto above=response(centered,mu,tau,middle).sum()>target;
          lo=at::where(above,middle,lo);hi=at::where(above,hi,middle);
        }
        auto lam=lo+(hi-lo)/2;c=response(centered,mu,tau,lam);
        auto free=(c>0)&(c<mu);
        if(free.any().item<bool>()){
          auto capped=(c==mu)&~free;
          lam=(centered.index({free}).sum()+tau*(mu.index({capped}).sum()-target))/free.sum();
          c=response(centered,mu,tau,lam);
        }
        result.multiplier=lam+shift;
        response_input=centered-lam;
      }
      result.controls=c;
    }
    auto c=result.controls;result.free=(c>0)&(c<mu);result.residual=zero;
    if(result.constraint!="zero"&&result.constraint!="caps"){
      if(((mu>0)&(response_input>0)&(c==0)).any().item<bool>())
        throw std::runtime_error("positive quadratic allocation support underflowed in requested dtype");
      result.residual=at::stack({(c-response(response_input,mu,tau,zero)).abs().max(),
                               (lower-c.sum()).clamp_min(0),(c.sum()-upper).clamp_min(0)}).max();
      const double eps=t.scalar_type()==at::kDouble?std::numeric_limits<double>::epsilon():std::numeric_limits<float>::epsilon();
      if(!at::isfinite(result.residual).item<bool>()||result.residual.item<double>()>64*t.numel()*eps)
        throw std::runtime_error("weighted quadratic allocation residual exceeds dtype tolerance");
    }
  }
  if(result.constraint=="caps")result.controls=mu;
  else if(result.constraint!="zero"){
    auto target=(result.constraint=="lower"||result.constraint=="equality")?lower:upper;
    // C++ custom Function records VariableInfo even for non-grad arguments;
    // its optional absent shift therefore needs a real scalar constant.
    auto bound_shift=shift.defined()?shift:at::zeros({},input.options());
    result.controls=ExactQuad::apply(input,mu,target,tau,result.controls,result.multiplier,result.constraint!="slack",bound_shift,response_input);
  }
  return result;
}
}  // namespace tide::weighted
