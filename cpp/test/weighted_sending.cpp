// Allocation §4.2–4.4/§6.3 literal standalone LibTorch witnesses.
#include "tide/weighted_sending.h"
#include <torch/csrc/autograd/autograd.h>
#include <cmath>
#include <stdexcept>

namespace {
using namespace tide;
void close(const Tensor& a,const Tensor& b){
  if(!at::allclose(a,b,a.scalar_type()==at::kDouble?1e-8:1e-5,a.scalar_type()==at::kDouble?1e-10:1e-6))
    throw std::runtime_error("standalone weighted sending mismatch");
}
}
void check_weighted_sending(const at::TensorOptions& o){
  using namespace tide;using namespace tide::weighted;
  auto s=at::tensor({0.,std::log(2.),std::log(3.)},o).set_requires_grad(true);
  auto a=at::tensor({1.,0.,1.},o).set_requires_grad(true),tau=at::ones({},o).set_requires_grad(true);
  auto p=attention_shares(s,a,tau);close(p,at::tensor({.25,0.,.75},o));
  auto gs=torch::autograd::grad({p[1]},{s,a,tau});
  close(gs[0],at::zeros_like(s));close(gs[1],at::tensor({0.,.5,0.},o));close(gs[2],at::zeros_like(tau));
  auto t=at::tensor({1.,1.,1.,0.},o),mu=at::full({4},.3,o).set_requires_grad(true),b=at::ones({},o).set_requires_grad(true);
  auto c=total_zero(t,mu,b,{3,2,1,0});close(c,at::tensor({.3,.3,.3,.1},o));
  auto g=torch::autograd::grad({c[3]},{mu,b});close(g[0],at::tensor({-1.,-1.,-1.,0.},o));close(g[1],at::ones({},o));
  close(hard_selection(t,mu,1,{3,2,1,0}),at::tensor({0.,0.,.3,0.},o));
  close(proportional(mu,1),at::full({4},.25,o));
  auto scores=at::tensor({std::log(9.),0.},o),caps=at::full({2},.1,o);
  close(attention_send(scores,caps,caps,"pooled_attention","mu",at::ones({},o)),at::tensor({.18,.02},o));
}
