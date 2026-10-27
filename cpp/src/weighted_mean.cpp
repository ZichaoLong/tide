#include "weighted_mean.h"
#include <torch/csrc/autograd/custom_function.h>
namespace tide::weighted {
namespace {
class Mean:public torch::autograd::Function<Mean> {
 public:
  static Tensor forward(torch::autograd::AutogradContext* ctx,Tensor mass,Tensor values){
    ctx->set_materialize_grads(false);auto total=mass.sum(-1);
    auto denominator=at::where(total>0,total,at::ones_like(total));
    auto result=(mass.unsqueeze(-1)*values).sum(-2)/denominator.unsqueeze(-1);
    ctx->save_for_backward({mass,values,total,result});return result;
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,torch::autograd::variable_list bars){
    if(at::GradMode::is_enabled())throw std::invalid_argument("weighted mean supports first-order VJP only");
    if(!bars[0].defined())return std::vector<Tensor>(2);
    auto s=ctx->get_saved_variables();auto mass=s[0],values=s[1],total=s[2],mean=s[3];
    auto denominator=at::where(total>0,total,at::ones_like(total));
    auto dm=((values-mean.unsqueeze(-2))*bars[0].unsqueeze(-2)).sum(-1)/denominator.unsqueeze(-1);
    if(mass.size(-1)==1)dm=at::zeros_like(mass);
    auto dv=(mass/denominator.unsqueeze(-1)).unsqueeze(-1)*bars[0].unsqueeze(-2);
    return {dm,dv};
  }
};
}
Tensor weighted_mean(const Tensor& mass,const Tensor& values){return Mean::apply(mass,values);}
}
