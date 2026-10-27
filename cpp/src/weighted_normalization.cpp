// Architecture §1: unchanged epsilon RMS map, normalized exact first-order VJP.
#include "tide/weighted.h"
#include <torch/csrc/autograd/custom_function.h>
namespace tide::weighted {
namespace {
class Normalize:public torch::autograd::Function<Normalize> {
 public:
  static Tensor forward(torch::autograd::AutogradContext* ctx,Tensor value,double epsilon){
    ctx->set_materialize_grads(false);auto denominator=at::sqrt(epsilon+value.square().mean(-1,true));
    auto normalized=value/denominator;ctx->save_for_backward({value,normalized,denominator});return normalized;
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,torch::autograd::variable_list bars){
    if(at::GradMode::is_enabled())throw std::invalid_argument("weighted normalization supports first-order VJP only");
    if(!bars[0].defined())return std::vector<Tensor>(2);
    auto s=ctx->get_saved_variables();auto n=s[1],denominator=s[2];
    return {(bars[0]-n*(bars[0]*n).mean(-1,true))/denominator,Tensor{}};
  }
};
}
Tensor normalize(const Tensor& x,double epsilon){return Normalize::apply(x,epsilon);}
}
