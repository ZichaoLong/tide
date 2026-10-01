#pragma once
#include <ATen/ATen.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tide::device_online::test {
// Independent FP64 formulas on the exact stored input. Near-constant LayerNorm
// is ill-conditioned: a lower-precision CPU result is not an exact oracle.
// This component-only engineering error budget does not relax graph equality.
struct LhPrecision {
  int64_t normalized_rows=0, strict_misses=0;
  double max_cpu_error=0,max_device_error=0,max_budget_fraction=0;
  void check(int64_t kind,const at::Tensor& input,const at::Tensor& weight,const at::Tensor& bias,
             const at::Tensor& cpu,const at::Tensor& device) {
    if(!at::isfinite(cpu).all().item<bool>()||!at::isfinite(device).all().item<bool>())
      throw std::runtime_error("nonfinite LH Full output");
    const auto act=(kind-1)/3,norm=(kind-1)%3;
    const bool half=input.scalar_type()==at::kHalf;
    if(!norm) {
      if(!at::allclose(device,cpu,half?3e-3:1e-5,half?2e-3:1e-6))throw std::runtime_error("LH activation differs from CPU");
      return;
    }
    ++normalized_rows;
    if(!at::allclose(device,cpu,1e-5,1e-6))++strict_misses;
    auto x=input.to(at::kDouble),w=weight.to(at::kDouble),b=bias.to(at::kDouble);
    if(act==0)x=at::clamp_min(x,0);else if(act==1)x=x/(1+at::exp(-x));
    auto centered=norm==2?x-x.mean():x;
    auto denominator=at::sqrt((centered*centered).mean()+(norm==1?1e-7:1e-5));
    auto normalized=centered/denominator,expected=normalized*w;
    if(norm==2)expected=expected+b;
    // Reduction depth and sensitivity to input/mean rounding. This is a
    // declared finite-fixture error estimate, not an all-input error theorem.
    const double depth=1+std::ceil(std::log2(std::max<int64_t>(1,x.numel())));
    const double u=half?1./1024:std::numeric_limits<float>::epsilon();
    auto budget=(half?2e-3:1e-6)+(half?3e-3:1e-5)*expected.abs()+4*u*depth*x.abs().max()/denominator*w.abs()*(1+normalized.abs());
    const auto ce=(cpu.to(at::kDouble)-expected).abs(),de=(device.to(at::kDouble)-expected).abs();
    max_cpu_error=std::max(max_cpu_error,ce.max().item<double>());
    max_device_error=std::max(max_device_error,de.max().item<double>());
    max_budget_fraction=std::max(max_budget_fraction,at::maximum(ce,de).div(budget).max().item<double>());
    if(!(ce<=budget).all().item<bool>()||!(de<=budget).all().item<bool>())
      throw std::runtime_error("LH norm exceeds independent FP64 conditioning budget");
  }
};
} // namespace tide::device_online::test
