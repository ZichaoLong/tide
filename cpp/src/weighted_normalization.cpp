// Architecture §1: unchanged epsilon RMS map, normalized exact first-order VJP.
#include "weighted_detail.h"
#include <torch/csrc/autograd/custom_function.h>
namespace tide::weighted {
namespace {
class Normalize:public torch::autograd::Function<Normalize> {
 public:
  static Tensor forward(torch::autograd::AutogradContext* ctx,Tensor value,double epsilon){
    ctx->set_materialize_grads(false);auto x=value.scalar_type()==at::kHalf?value.to(at::kFloat):value;
    auto denominator=at::sqrt(epsilon+x.square().mean(-1,true));
    auto normalized=x/denominator;ctx->save_for_backward({value,normalized,denominator});return normalized.to(value.scalar_type());
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,torch::autograd::variable_list bars){
    if(at::GradMode::is_enabled())throw std::invalid_argument("weighted normalization supports first-order VJP only");
    if(!bars[0].defined())return std::vector<Tensor>(2);
    auto s=ctx->get_saved_variables();auto n=s[1],denominator=s[2];auto bar=bars[0];
    if(s[0].scalar_type()==at::kHalf){bar=bar.to(at::kFloat);n=s[0].to(at::kFloat)/denominator;}
    return {((bar-n*(bar*n).mean(-1,true))/denominator).to(s[0].scalar_type()),Tensor{}};
  }
};
class HalfScaledRMS:public torch::autograd::Function<HalfScaledRMS> {
 public:
  static Tensor forward(torch::autograd::AutogradContext* ctx,Tensor value,Tensor scale,double epsilon){
    ctx->set_materialize_grads(false);auto x=value.to(at::kFloat);
    auto denominator=at::sqrt(epsilon+x.square().mean(-1,true));ctx->save_for_backward({value,scale,denominator});
    return (scale.to(at::kFloat)*x/denominator).to(value.scalar_type());
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,torch::autograd::variable_list bars){
    if(at::GradMode::is_enabled())throw std::invalid_argument("weighted scaled RMS supports first-order VJP only");
    if(!bars[0].defined())return std::vector<Tensor>(3);
    auto s=ctx->get_saved_variables();auto n=s[0].to(at::kFloat)/s[2],bar=bars[0].to(at::kFloat);
    auto weighted=bar*s[1].to(at::kFloat);
    auto dx=(weighted-n*(weighted*n).mean(-1,true))/s[2];
    auto ds=(bar*n).sum_to_size(s[1].sizes());
    return {dx.to(s[0].scalar_type()),ds.to(s[1].scalar_type()),Tensor{}};
  }
};
class HalfL2:public torch::autograd::Function<HalfL2> {
 public:
  static Tensor forward(torch::autograd::AutogradContext* ctx,Tensor value,double epsilon){
    ctx->set_materialize_grads(false);auto x=value.to(at::kFloat);
    auto denominator=at::sqrt(epsilon+x.square().sum(-1,true));ctx->save_for_backward({value,denominator});
    return (x/denominator).to(value.scalar_type());
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,torch::autograd::variable_list bars){
    if(at::GradMode::is_enabled())throw std::invalid_argument("weighted epsilon L2 supports first-order VJP only");
    if(!bars[0].defined())return std::vector<Tensor>(2);
    auto s=ctx->get_saved_variables();auto n=s[0].to(at::kFloat)/s[1],bar=bars[0].to(at::kFloat);
    return {((bar-n*(bar*n).sum(-1,true))/s[1]).to(s[0].scalar_type()),Tensor{}};
  }
};
}
Tensor normalize(const Tensor& x,double epsilon){return Normalize::apply(x,epsilon);}
namespace detail {
Tensor scaled_rms(const Tensor& value,const Tensor& scale,double epsilon){
  if(value.scalar_type()==at::kHalf)return HalfScaledRMS::apply(value,scale,epsilon);
  return scale*value/at::sqrt(epsilon+value.square().mean(-1,true));
}
Tensor epsilon_l2(const Tensor& value,double epsilon){
  if(value.scalar_type()==at::kHalf)return HalfL2::apply(value,epsilon);
  return value/at::sqrt(epsilon+value.square().sum(-1,true));
}
}
}
