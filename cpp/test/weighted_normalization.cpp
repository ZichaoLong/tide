// Architecture §1: independent epsilon RMS Jacobian, unchanged forward.
#include "tide/weighted.h"
#include <torch/csrc/autograd/autograd.h>
#include <cmath>
#include <stdexcept>
void check_weighted_normalization(const at::TensorOptions& o){
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
