// Architecture §1: independent epsilon RMS Jacobian, unchanged forward.
#include "tide/weighted.h"
#include "../src/weighted_detail.h"
#include <torch/csrc/autograd/autograd.h>
#include <cmath>
#include <stdexcept>
namespace {
void half_literal(){
  using namespace tide::weighted::detail;
  auto old=at::full({},(tide::Index{1}<<54)+7,at::kLong),now=old+100000;
  auto decay=passive_decay(now-old,{0.,1e-6},at::zeros({2},at::kHalf));
  if(!at::equal(decay,at::tensor({1.,std::exp(-.1)},at::kFloat).to(at::kHalf)))
    throw std::runtime_error("half long-clock zero-rate decay mismatch");
  auto x=at::tensor({0.,0.,1e-4,-2e-4,3.,4.},at::kFloat).to(at::kHalf).reshape({3,2}).set_requires_grad(true);
  auto scale=at::tensor({.3,-.7},at::kFloat).to(at::kHalf).set_requires_grad(true);
  auto bar=at::tensor({.4,-.2,.6,.8,-.3,.5},at::kFloat).to(at::kHalf).reshape({3,2});
  for(int kind=0;kind<3;++kind){
    auto y=kind==2?tide::weighted::detail::scaled_rms(x,scale,1e-5):
      kind==1?tide::weighted::detail::epsilon_l2(x,1e-5):tide::weighted::normalize(x,1e-5);
    auto expected=at::zeros_like(x),dx=at::zeros_like(x),ds=at::zeros({2},at::kDouble);
    const double divisor=kind==1?1.:2.;
    for(int i=0;i<3;++i){double v[2]={x[i][0].item<double>(),x[i][1].item<double>()};
      const double den=std::sqrt(1e-5+(v[0]*v[0]+v[1]*v[1])/divisor);
      for(int j=0;j<2;++j){double gradient=0.;expected[i][j].fill_((kind==2?scale[j].item<double>():1.)*v[j]/den);
        for(int k=0;k<2;++k)gradient+=bar[i][k].item<double>()*(kind==2?scale[k].item<double>():1.)*
          ((j==k?1.:0.)/den-v[j]*v[k]/(divisor*den*den*den));
        dx[i][j].fill_(gradient);ds[j].add_(bar[i][j].item<double>()*v[j]/den);
      }
    }
    auto g=torch::autograd::grad({y},{x,scale},{bar},false,false,true);
    if(!at::allclose(y,expected,2e-3,1e-3)||!at::allclose(g[0],dx,2e-3,1e-3)||
       (kind==2?!at::allclose(g[1],ds.to(at::kHalf),2e-3,1e-3):g[1].defined()))
      throw std::runtime_error("half normalization literal Jacobian mismatch");
  }
}
}
void check_weighted_normalization(const at::TensorOptions& o){
  if(o.dtype()==at::kFloat)half_literal();
  auto x=at::tensor({3.,4.,-2.,.5},o).reshape({2,2}).set_requires_grad(true);
  auto y=tide::weighted::normalize(x,2.);
  if(!at::equal(y,x/at::sqrt(2.+x.square().mean(-1,true))))throw std::runtime_error("RMS forward changed");
  for(tide::Index row=0;row<2;++row)for(tide::Index column=0;column<2;++column){
    std::array<double,2> v=row?std::array<double,2>{-2.,.5}:std::array<double,2>{3.,4.};
    double den=std::sqrt(2+(v[0]*v[0]+v[1]*v[1])/2);auto expected=at::zeros_like(x);
    for(tide::Index j=0;j<2;++j)expected[row][j].fill_((j==column?1.:0.)/den-v[j]*v[column]/(2*den*den*den));
    auto grad=torch::autograd::grad({y[row][column]},{x},{},true)[0];
    if(!at::allclose(grad,expected,o.dtype()==at::kDouble?1e-8:1e-5,o.dtype()==at::kDouble?1e-10:1e-6))
      throw std::runtime_error("RMS literal Jacobian mismatch");
  }
}
