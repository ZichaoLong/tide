#include "weighted_attention_mean.h"
#include <torch/csrc/autograd/custom_function.h>
namespace tide::weighted {
namespace {
class AttentionMean:public torch::autograd::Function<AttentionMean> {
 public:
  static Tensor forward(torch::autograd::AutogradContext* ctx,Tensor logits,Tensor values){
    ctx->set_materialize_grads(false);auto probabilities=logits.softmax(-1);
    ctx->save_for_backward({logits,values,probabilities});
    return (probabilities.unsqueeze(-1)*values).sum(-2);
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,torch::autograd::variable_list bars){
    if(at::GradMode::is_enabled())throw std::invalid_argument("weighted Attention supports first-order VJP only");
    if(!bars[0].defined())return std::vector<Tensor>(2);
    auto s=ctx->get_saved_variables();auto values=s[1],p=s[2];
    auto delta=values-values.slice(-2,0,1);
    auto centered=delta-(p.unsqueeze(-1)*delta).sum(-2,true);
    return {p*(centered*bars[0].unsqueeze(-2)).sum(-1),p.unsqueeze(-1)*bars[0].unsqueeze(-2)};
  }
};
}
Tensor attention_mean(const Tensor& logits,const Tensor& values){return AttentionMean::apply(logits,values);}
}
