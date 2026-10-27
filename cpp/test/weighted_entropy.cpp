// Allocation §5.6/§6.2: standalone analytical Bern and Shannon witnesses.
#include "tide/weighted.h"
#include <torch/csrc/autograd/autograd.h>
#include <stdexcept>

namespace {
using namespace tide;
void close(const Tensor& a,const Tensor& b){
  if(!at::allclose(a,b,a.scalar_type()==at::kDouble?1e-8:1e-5,a.scalar_type()==at::kDouble?1e-10:1e-6))
    throw std::runtime_error("standalone entropy formula mismatch");
}
}
void check_weighted_entropy(const at::TensorOptions& o){
  using namespace tide;using namespace tide::weighted;
  auto t=at::zeros({2},o).set_requires_grad(true),mu=at::tensor({.4,.6},o).set_requires_grad(true);
  auto total=at::full({},.25,o).set_requires_grad(true),tau=at::full({},.7,o).set_requires_grad(true);
  auto result=allocate(t,mu,total,total,tau,"bern");
  close(result.controls,at::tensor({.1,.15},o));
  auto grads=torch::autograd::grad({result.controls[0]},{t,mu,total,tau});
  close(grads[0],at::tensor({.06428571428571429,-.06428571428571429},o));
  close(grads[1],at::tensor({.15,-.1},o));close(grads[2],at::full({},.4,o));close(grads[3],at::zeros({},o));
  auto scores=at::tensor({.2,-.1,.4},o),caps=at::tensor({.8,.7,.9},o),one=at::ones({},o);
  auto shannon=allocate(scores,caps,one,one,tau,"ent",1.);
  close(shannon.controls,at::softmax(scores/tau,0));
  auto ent2=allocate(scores,caps,at::full({},.2,o),at::full({},2.,o),tau,"ent",2.);
  close(ent2.controls,allocate_quad(scores+tau/2,caps,at::full({},.2,o),at::full({},2.,o),tau).controls);
}
