// Analytic copy/cast VJP; metadata and unrelated roots are independent.
#include "../src/weighted_placement.h"
#include <torch/csrc/autograd/autograd.h>
#include <ATen/core/grad_mode.h>
#include <stdexcept>
void check_weighted_placement(const at::TensorOptions& o){
  using namespace tide;
  auto a=at::full({2},.25,o).set_requires_grad(true);
  auto b=at::full({2},.5,o).set_requires_grad(true);
  auto c=at::full({2},.75,o).set_requires_grad(true);
  std::vector<Tensor> roots{a,b,c};auto alias=a;
  auto clock=at::tensor(std::vector<Index>{(Index{1}<<54)+3,-1},at::TensorOptions().dtype(at::kLong)).to(o.device());
  auto expected=clock.to(at::kCPU);
  weighted::placement_tensors({&a,&alias,&b,&c,&clock},at::Device(at::kCPU),o.dtype()==at::kFloat?at::kDouble:at::kFloat);
  if(a.unsafeGetTensorImpl()!=alias.unsafeGetTensorImpl()||clock.scalar_type()!=at::kLong||!at::equal(clock,expected))
    throw std::runtime_error("weighted placement lost alias/int64 metadata");
  auto loss=a.sum()+3*alias.sum()+0*b.sum();
  auto gradients=torch::autograd::grad({loss},roots,{},true,false,true);
  if(!at::equal(gradients[0],at::full_like(roots[0],4))||!at::equal(gradients[1],at::zeros_like(roots[1]))||gradients[2].defined())
    throw std::runtime_error("weighted placement merged independent cotangent roots");
  {at::NoGradGuard guard;roots[0].add_(1);}
  bool rejected=false;
  try{torch::autograd::grad({loss},roots,{},false,false,true);}catch(const std::exception&){rejected=true;}
  if(!rejected)throw std::runtime_error("weighted placement lost original version guard");
}
