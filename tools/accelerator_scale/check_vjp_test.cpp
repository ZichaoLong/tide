#include "check_vjp.h"
#include <ATen/Parallel.h>
#include <iostream>
#include <stdexcept>

namespace {
using accelerator_scale::check_vjp;
using tide::Tensor;
Tensor leaf() {
  return at::tensor({.001f,.002f,-.003f,.004f,.005f,-.006f,.007f,.008f}).set_requires_grad(true);
}
Tensor full(const Tensor& x) {return at::rms_norm(at::silu(x),{8},{},1e-7);}
template<class Fn> void rejects(Fn fn,const std::string& reason) {
  try {fn();}catch(const std::runtime_error& error) {
    if(std::string(error.what()).find(reason)!=std::string::npos)return;
    throw;
  }
  throw std::runtime_error("VJP gate accepted injected error: "+reason);
}
}
int main() {
  try {
    at::set_num_threads(1);at::set_num_interop_threads(1);
    auto x=leaf(),y=leaf(),unused=leaf(),unused_y=leaf();
    for(bool zero:{false,true})for(bool conditioned:{false,true})
      check_vjp(full(x),full(y),{x,unused},{y,unused_y},0,zero,conditioned);

    // Identical forward values with a deliberately wrong declared derivative:
    // the conditioned policy must reject its Jacobian, not excuse cancellation.
    auto z=full(y);auto wrong=z.detach()+1.01*(z-z.detach());
    rejects([&]{check_vjp(full(x),wrong,{x},{y},0,false,false);},"placement changed VJP");
    rejects([&]{check_vjp(full(x),wrong,{x},{y},0,false,true);},"basis=");

    // A zero connection is observably different from an unused owner, even
    // for an explicitly zero objective and the conditioned policy.
    rejects([&]{check_vjp(full(x),full(y)+unused_y.sum()*0,{x,unused},{y,unused_y},0,true,true);},"None connectivity");
    rejects([&]{check_vjp(x,y*1e38,{x},{y},0,false,true);},"finiteness");
    std::cout << "VJP policy identity, wrong Jacobian, None/zero and nonfinite gates passed\n";
    return 0;
  }catch(const std::exception& error){std::cerr << error.what() << '\n';return 1;}
}
