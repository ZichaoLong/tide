// Literal complete rows: quad exact solution/VJP and fractional stable fill.
#include "tide/weighted.h"
#include "../src/weighted_sending_batch.h"
#include "../src/weighted_mean.h"
#include "../src/region_metadata.h"
#include <torch/csrc/autograd/autograd.h>
#include <limits>
#include <stdexcept>
namespace {
using namespace tide;
void close(const Tensor& a,const Tensor& b){
  if(!at::allclose(a,b,a.scalar_type()==at::kDouble?1e-8:1e-5,a.scalar_type()==at::kDouble?1e-10:1e-6))
    throw std::runtime_error("standalone weighted region batch mismatch");
}
}
void check_weighted_region_batch(const at::TensorOptions& o){
  using namespace tide;using namespace tide::weighted;
  // Literal metadata witnesses exercise the grouped reader even on CPU.
  std::vector<Tensor> controls;
  for(double value:{0.,1.,-1.,std::numeric_limits<double>::quiet_NaN(),
                    std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity()})
    controls.push_back(at::full({},value,o).set_requires_grad(true));
  if(region_metadata::group_flags(controls,true)!=std::vector<bool>{false,true,false,false,true,false}
    ||region_metadata::group_flags(controls,false)!=std::vector<bool>{true,true,true,false,false,false})
    throw std::runtime_error("grouped region metadata differs from literal support/finite flags");
  if(region_metadata::group_flags({at::empty({0},o),at::ones({2},o),at::full({2},std::numeric_limits<double>::infinity(),o)},false)
      !=std::vector<bool>{true,true,false})throw std::runtime_error("vector/empty region finite metadata differs");
  auto metadata_roots=torch::autograd::grad({3*controls[0]+0*controls[1]},
      {controls[0],controls[1],controls[2]},{},true,false,true);
  close(metadata_roots[0],at::full({},3.,o));close(metadata_roots[1],at::zeros({},o));
  if(metadata_roots[2].defined())throw std::runtime_error("metadata connected an unused control root");
  auto singleton=at::tensor({.35},o).set_requires_grad(true),content=at::tensor({.7,-.4},o).reshape({1,2}).set_requires_grad(true);
  auto mean=weighted_mean(singleton,content);auto bar=at::tensor({1.e5,-2.e5},o);
  auto mean_grad=torch::autograd::grad({mean},{singleton,content},{bar});
  close(mean_grad[0],at::zeros_like(singleton));close(mean_grad[1],bar.unsqueeze(0));
  auto s=at::tensor({2.,1.,100.},o).set_requires_grad(true),mu=at::tensor({1.,1.,0.},o).set_requires_grad(true);
  auto total=at::full({},1.4,o).set_requires_grad(true),tau=at::full({},2.,o).set_requires_grad(true);
  auto unused=at::tensor({-1.,3.,0.},o).set_requires_grad(true);
  auto result=allocate_batch({{s,mu,total,total,tau,{}},{unused,mu,mu.sum(),mu.sum(),tau,{}}},"quad");
  close(result[0].controls,at::tensor({.95,.45,0.},o));close(result[1].controls,mu);
  auto grads=torch::autograd::grad({result[0].controls[0]},{s,mu,total,tau,unused},{},true,false,true);
  close(grads[0],at::tensor({.25,-.25,0.},o));close(grads[1],at::zeros_like(mu));
  close(grads[2],at::full({},.5,o));close(grads[3],at::full({},-.125,o));
  if(grads[4].defined())throw std::runtime_error("batch allocation connected an unused row");
  auto cap_grads=torch::autograd::grad({result[1].controls.sum()},{unused,mu},{},true,false,true);
  if(cap_grads[0].defined())throw std::runtime_error("caps branch connected scores");close(cap_grads[1],at::ones_like(mu));
  auto scores=at::tensor({.2,.2,-1.,-.5,-1.,-.5},o).reshape({2,3});
  auto caps=at::tensor({.4,.7,.5,.5,.2,.8},o).reshape({2,3});
  auto ids=at::tensor({3,1,7,7,4,2},o.dtype(at::kLong)).reshape({2,3});
  auto fill=batch::sorted_send(scores,caps,ids,at::tensor({.9,.3},o));
  close(fill,at::tensor({.2,.7,0.,0.,0.,.3},o).reshape({2,3}));
  auto hard=batch::sorted_send(scores,caps,ids,at::zeros({2},o),1);
  close(hard,at::tensor({0.,.7,0.,0.,0.,.8},o).reshape({2,3}));
}
