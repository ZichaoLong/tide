// Independent bounded bisection; backward differentiates the exact Alloc map.
#include "weighted_regularizers.h"
#include "tide/allocation_observation.h"
#include <torch/csrc/autograd/custom_function.h>
#include <limits>

namespace tide::weighted {
using namespace detail;
namespace {
class ExactEntropy:public torch::autograd::Function<ExactEntropy> {
 public:
  static Tensor forward(torch::autograd::AutogradContext* ctx,Tensor scores,Tensor caps,Tensor target,
                        Tensor temperature,Tensor c,Tensor a,Tensor e,Tensor dtau,bool equality,Tensor shift){
    ctx->set_materialize_grads(false);ctx->save_for_backward({scores,caps,target,temperature,a,e,dtau,shift});
    ctx->saved_data["equality"]=equality;return c.clone();
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,torch::autograd::variable_list grads){
    if(!grads[0].defined())return std::vector<Tensor>(10);
    at::AutoGradMode first_order(false);auto s=ctx->get_saved_variables();auto a=s[4],e=s[5],dtau=s[6],total=a.sum();
    const bool equality=ctx->saved_data.at("equality").toBool();auto mean=at::zeros({},a.options()),centered=grads[0];
    if(equality&&total.item<double>()>0){auto anchor=grads[0][a.argmax().item<Index>()],relative=grads[0]-anchor,offset=(a*relative).sum()/total;mean=anchor+offset;centered=relative-offset;}
    auto gt=a*centered,shift=s[7];auto gs=shift.defined()?(equality?at::zeros({},a.options()):-gt.sum()):Tensor{};
    return {gt,e*centered,mean,(dtau*centered).sum(),{},{},{},{},{},gs};
  }
};
void validate(const Tensor& t,const Tensor& mu,const Tensor& lo,const Tensor& hi,const Tensor& tau){
  require(t.defined()&&t.dim()==1&&t.numel()>0&&(t.scalar_type()==at::kFloat||t.scalar_type()==at::kDouble),"allocation requires FP32/FP64 scores");
  for(const auto& x:{mu,lo,hi,tau})require(x.defined()&&x.scalar_type()==t.scalar_type()&&x.device()==t.device()&&at::isfinite(x).all().item<bool>(),"allocation dtype/device/value mismatch");
  require(mu.sizes()==t.sizes()&&lo.dim()==0&&hi.dim()==0&&tau.dim()==0&&at::isfinite(t).all().item<bool>(),"invalid allocation shape/scores");
  require((mu>=0).all().item<bool>()&&(mu<=1).all().item<bool>()&&lo.item<double>()>=0&&hi.item<double>()>=lo.item<double>()
    &&lo.item<double>()<=mu.sum().item<double>()&&tau.item<double>()>0,"infeasible allocation bounds/temperature");
}
}
Allocation allocate(const Tensor& input,const Tensor& mu,const Tensor& lower,const Tensor& upper,const Tensor& tau,
                    const std::string& regularizer,std::optional<double> alpha,const Tensor& shift){
  auto t=input;
  Regularizer rule(regularizer,alpha);
  if(regularizer=="quad")return allocate_quad(t,mu,lower,upper,tau,shift);
  validate(t,mu,lower,upper,tau);Allocation result;Tensor x,a,e,dtau;
  const AllocationRow row{input,mu,lower,upper,tau,shift};
  auto failure=[&](const std::string& message){return scalar_allocation_failure(message,row,result,x,regularizer,alpha);};
  if(shift.defined())require(shift.dim()==0&&shift.device()==t.device()&&shift.scalar_type()==t.scalar_type()
    &&at::isfinite(shift).item<bool>(),"invalid allocation score_shift");
  result.backward=regularizer+"-exact-map-v1";
  {
    at::NoGradGuard guard;
    if(shift.defined()){t=input-shift;require(at::isfinite(t).all().item<bool>(),"shifted allocation scores exceed requested dtype range");}
    auto mass=mu.sum(),zero=at::zeros({},t.options());result.multiplier=zero;
    if(mass.item<double>()==0||upper.item<double>()==0){result.controls=at::zeros_like(t);result.constraint="zero";}
    else if(lower.item<double>()==mass.item<double>()){result.controls=mu.clone();result.constraint="caps";}
    else {
      x=t;auto c=rule.response(x,mu,tau);auto sum=c.sum().item<double>();
      result.constraint=sum<lower.item<double>()?"lower":sum>upper.item<double>()?"upper":"slack";
      if(lower.item<double>()==upper.item<double>())result.constraint="equality";
      if(result.constraint!="slack"){
        auto target=(result.constraint=="lower"||result.constraint=="equality")?lower:upper;
        auto origin=input.max(),centered=input-origin;auto points=centered-rule.inverse(target/mass,mu,tau);
        auto positive=mu>0;auto low=points.index({positive}).min(),high=points.index({positive}).max();
        for(int i=0;i<80;++i){
          auto middle=low+(high-low)/2;
          if(i%8==0&&((middle==low)|(middle==high)).item<bool>())break;
          auto total=rule.response(centered-middle,mu,tau).sum();
          auto above=total>target;low=at::where(above,middle,low);high=at::where(above,high,middle);
        }
        auto middle=low+(high-low)/2;std::vector<Tensor> choices{low,middle,high},errors;
        for(const auto& value:choices)errors.push_back((rule.response(centered-value,mu,tau).sum()-target).abs());
        auto lam=choices[at::stack(errors).argmin().item<Index>()];x=centered-lam;c=rule.response(x,mu,tau);
        result.multiplier=lam+origin;if(shift.defined())result.multiplier=result.multiplier-shift;
      }
      result.controls=c;
    }
    auto c=result.controls;result.free=(c>0)&(c<mu);result.residual=zero;
    if(result.constraint!="zero"&&result.constraint!="caps"){
      result.residual=at::stack({(c-rule.response(x,mu,tau)).abs().max(),(lower-c.sum()).clamp_min(0),(c.sum()-upper).clamp_min(0)}).max();
      const double eps=t.scalar_type()==at::kDouble?std::numeric_limits<double>::epsilon():std::numeric_limits<float>::epsilon();
      if(!at::isfinite(result.residual).item<bool>()||result.residual.item<double>()>64*t.numel()*eps)
        throw failure("entropy allocation residual exceeds dtype tolerance");
      if((rule.support(x,mu,tau)&(c==0)).any().item<bool>())throw failure("positive allocation support underflowed in requested dtype");
      auto slopes=rule.slopes(x,mu,tau,c);a=slopes[0];e=slopes[1];dtau=slopes[2];
    }
  }
  if(result.constraint=="caps")result.controls=mu;
  else if(result.constraint!="zero"){
    auto target=(result.constraint=="lower"||result.constraint=="equality")?lower:upper;
    auto bound_shift=shift.defined()?shift:at::zeros({},input.options());
    result.controls=ExactEntropy::apply(input,mu,target,tau,result.controls,a,e,dtau,result.constraint!="slack",bound_shift);
  }
  observe_allocation(row,result,regularizer,alpha,"native-scalar");
  return result;
}
}
