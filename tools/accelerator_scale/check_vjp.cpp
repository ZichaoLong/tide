#include "check_vjp.h"
#include <torch/csrc/autograd/autograd.h>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace accelerator_scale {
namespace {
constexpr double atol=1e-6,rtol=1e-5;
Tensor cpu(const Tensor& t) {return t.detach().to(at::kCPU);}
void structure(const Tensor& a,const Tensor& b,size_t root,size_t owner) {
  if(a.defined()!=b.defined())
    throw std::runtime_error("placement changed None connectivity: root="+std::to_string(root)+" owner="+std::to_string(owner));
  if(a.defined() && (a.sizes()!=b.sizes() || a.scalar_type()!=b.scalar_type()
      || !at::isfinite(a).all().item<bool>() || !at::isfinite(b).all().item<bool>()))
    throw std::runtime_error("placement changed VJP metadata or finiteness");
}
std::string error(const Tensor& a,const Tensor& b,size_t root,size_t owner) {
  const auto diff=(b-a).abs();
  std::ostringstream s;
  s << "placement changed VJP: root=" << root << " owner=" << owner << " shape=" << a.sizes()
    << " max_abs=" << diff.max().item<double>() << " reference_max=" << a.abs().max().item<double>()
    << " max_tolerance_ratio=" << (diff/(atol+rtol*a.abs())).max().item<double>();
  return s.str();
}
struct Contraction {Tensor expected,actual,expected_scale,actual_scale;};
}
void check_vjp(const Tensor& expected,const Tensor& actual,
               const std::vector<Tensor>& expected_leaves,const std::vector<Tensor>& actual_leaves,
               size_t root,bool zero,bool conditioned) {
  auto a=torch::autograd::grad({expected.square().sum()*(zero?0.:.7)},expected_leaves,{},true,false,true);
  auto b=torch::autograd::grad({actual.square().sum()*(zero?0.:.7)},actual_leaves,{},true,false,true);
  std::vector<size_t> failures;
  for(size_t i=0;i<a.size();++i) {
    structure(a[i],b[i],root,i);
    if(!a[i].defined())continue;
    a[i]=cpu(a[i]);b[i]=cpu(b[i]);
    if(!at::allclose(b[i],a[i],rtol,atol)) failures.push_back(i);
  }
  if(failures.empty())return;
  if(!conditioned || zero || expected.numel()>64)
    throw std::runtime_error(error(a[failures[0]],b[failures[0]],root,failures[0]));

  // Explicit diagnostic policy for a cancellation-sensitive contraction. Check
  // every column of the root's VJP (the complete Jacobian), with the ORIGINAL
  // componentwise tolerances, then reconstruct both quadratic VJPs in FP64.
  // It never repairs a changed Jacobian, disconnected owner or nonfinite value.
  const auto x=cpu(expected).reshape({-1}).to(at::kDouble)*1.4;
  const auto y=cpu(actual).reshape({-1}).to(at::kDouble)*1.4;
  std::vector<Contraction> sums(a.size());
  for(size_t i=0;i<a.size();++i)if(a[i].defined()) {
    auto z=at::zeros_like(a[i],at::TensorOptions().dtype(at::kDouble));
    sums[i]={z.clone(),z.clone(),z.clone(),z.clone()};
  }
  double max_basis_ratio=0.;
  for(int64_t j=0;j<expected.numel();++j) {
    auto u=torch::autograd::grad({expected.reshape({-1})[j]},expected_leaves,{},true,false,true);
    auto v=torch::autograd::grad({actual.reshape({-1})[j]},actual_leaves,{},true,false,true);
    for(size_t i=0;i<u.size();++i) {
      structure(u[i],v[i],root,i);
      if(u[i].defined()!=a[i].defined())throw std::runtime_error("basis/quadratic owner connectivity differs");
      if(!u[i].defined())continue;
      auto left=cpu(u[i]),right=cpu(v[i]);
      if(!at::allclose(right,left,rtol,atol))
        throw std::runtime_error("basis="+std::to_string(j)+" "+error(left,right,root,i));
      max_basis_ratio=std::max(max_basis_ratio,((right-left).abs()/(atol+rtol*left.abs())).max().item<double>());
      auto e=left.to(at::kDouble)*x[j],c=right.to(at::kDouble)*y[j];
      auto& sum=sums[i];
      sum.expected+=e;sum.actual+=c;sum.expected_scale+=e.abs();sum.actual_scale+=c.abs();
    }
  }
  double max_contraction_ratio=0.;
  for(size_t i=0;i<a.size();++i)if(a[i].defined()) {
    const auto& sum=sums[i];
    // A cancellation-aware scale is sum(|J_ij * cotangent_j|), not the
    // possibly tiny final signed sum. This policy is explicit, benchmark-only.
    const auto e=(a[i].to(at::kDouble)-sum.expected).abs()/(atol+rtol*sum.expected_scale);
    const auto c=(b[i].to(at::kDouble)-sum.actual).abs()/(atol+rtol*sum.actual_scale);
    const auto ratio=std::max(e.max().item<double>(),c.max().item<double>());
    if(!at::isfinite(e).all().item<bool>() || !at::isfinite(c).all().item<bool>() || ratio>1.)
      throw std::runtime_error("quadratic VJP disagrees with complete basis contraction: owner="+std::to_string(i));
    max_contraction_ratio=std::max(max_contraction_ratio,ratio);
  }
  for(auto i:failures)std::cout << "NUMERICAL strict_quadratic_failed " << error(a[i],b[i],root,i) << '\n';
  std::cout << "NUMERICAL policy=basis-conditioned root=" << root << " coordinates=" << expected.numel()
            << " max_basis_tolerance_ratio=" << max_basis_ratio
            << " max_contraction_tolerance_ratio=" << max_contraction_ratio << " passed\n" << std::flush;
}
}
