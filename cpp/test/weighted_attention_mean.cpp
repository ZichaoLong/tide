// KV §3/4: independent literal Jacobian and exact constant-value invariance.
#include "../src/weighted_attention_mean.h"
#include <torch/csrc/autograd/autograd.h>
#include <cmath>
#include <stdexcept>

void check_weighted_attention_mean(const at::TensorOptions& o){
  auto close=[&](const at::Tensor& x,const at::Tensor& y){
    if(!at::allclose(x,y,o.dtype()==at::kDouble?1e-8:1e-5,o.dtype()==at::kDouble?1e-10:1e-6))
      throw std::runtime_error("weighted Attention literal VJP mismatch");
  };
  const double p=1/(1+std::exp(-.5));
  auto logits=at::tensor({.2,-.3},o).set_requires_grad(true);
  auto base=at::tensor({.25,-.5,.75,.5},o).reshape({2,2});
  auto bar=at::tensor({.3,-.7},o);at::Tensor previous;
  for(double shift:{0.,65536.}){
    auto values=(base+shift).set_requires_grad(true);
    auto result=tide::weighted::attention_mean(logits,values);
    if(!at::equal(result,(logits.softmax(-1).unsqueeze(-1)*values).sum(-2)))
      throw std::runtime_error("weighted Attention forward changed");
    auto grads=torch::autograd::grad({result},{logits,values},{bar});
    close(grads[0],at::tensor({p*(1-p)*.55,-p*(1-p)*.55},o));
    close(grads[1],at::tensor({p,1-p},o).unsqueeze(-1)*bar.unsqueeze(0));
    if(previous.defined()&&!at::equal(previous,grads[0]))throw std::runtime_error("weighted Attention lost value translation invariance");
    previous=grads[0];
  }
  for(tide::Index length:{1,4}){
    auto l=at::linspace(-.2,.5,length,o).set_requires_grad(true);
    auto v=at::full({length,3},713.,o).set_requires_grad(true);
    auto grads=torch::autograd::grad({tide::weighted::attention_mean(l,v).sum()},{l,v});
    if(!at::equal(grads[0],at::zeros_like(l)))throw std::runtime_error("constant values have nonzero logit gradient");
    close(grads[1],l.softmax(-1).unsqueeze(-1).expand_as(v));
  }
}
