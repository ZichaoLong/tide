#include "weighted_attention_mean.h"
#include <torch/csrc/autograd/custom_function.h>
namespace tide::weighted {
namespace {
class AttentionMean:public torch::autograd::Function<AttentionMean> {
 public:
  static Tensor forward(torch::autograd::AutogradContext* ctx,Tensor logits,Tensor values){
    ctx->set_materialize_grads(false);const bool half=values.scalar_type()==at::kHalf;
    auto probabilities=(half?logits.to(at::kFloat):logits).softmax(-1);
    ctx->save_for_backward({logits,values,probabilities});
    auto result=(probabilities.unsqueeze(-1)*(half?values.to(at::kFloat):values)).sum(-2);
    return half?result.to(values.scalar_type()):result;
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,torch::autograd::variable_list bars){
    if(at::GradMode::is_enabled())throw std::invalid_argument("weighted Attention supports first-order VJP only");
    if(!bars[0].defined())return std::vector<Tensor>(2);
    auto s=ctx->get_saved_variables();auto values=s[1],p=s[2],bar=bars[0];
    if(values.scalar_type()==at::kHalf){values=values.to(at::kFloat);bar=bar.to(at::kFloat);}
    auto delta=values-values.slice(-2,0,1);
    auto centered=delta-(p.unsqueeze(-1)*delta).sum(-2,true);
    return {(p*(centered*bar.unsqueeze(-2)).sum(-1)).to(s[0].scalar_type()),
      (p.unsqueeze(-1)*bar.unsqueeze(-2)).to(s[1].scalar_type())};
  }
};
}
Tensor attention_mean(const Tensor& logits,const Tensor& values){return AttentionMean::apply(logits,values);}
}
