#include "scoring.h"
#include <tide/read.h>
#include <tide/autograd.h>
#include <torch/csrc/autograd/autograd.h>
#include <iostream>
#include <stdexcept>

int main() {
  try {
    using namespace accelerator_scale;
    Scoring fp32; fp32.dtype = at::kFloat;
    pdg_scale::Fixture f;
    tide::Node node; node.readout = "norm-fp64-v1";
    f.graph.nodes = {node}; f.model.nodes.resize(1);
    configure_scoring(f, fp32);
    if (f.graph.nodes[0].readout != "scale-norm-fp32-v1") throw std::runtime_error("precision identity");
    auto& w=f.model.nodes[0];
    auto x=at::tensor({3.f,4.f}).set_requires_grad(true);
    tide::State state;state.value=x;
    tide::ReadInput input{&state,0,{x,{},{},{},-1}};
    auto value=w.read_kernel->step(w,input);
    if (value.scalar_type()!=at::kFloat || value.item<float>()!=5.f) throw std::runtime_error("analytic norm");
    auto gradient=torch::autograd::grad({value},{x})[0];
    if (!at::allclose(gradient,at::tensor({.6f,.8f}),1e-6,1e-7)) throw std::runtime_error("analytic norm VJP");
    auto zero=at::zeros({2},x.options()).set_requires_grad(true);state.value=zero;
    auto z=w.read_kernel->step(w,{&state,0,{zero,{},{},{},-1}});
    if (!at::equal(torch::autograd::grad({z},{zero})[0],at::zeros_like(zero))) throw std::runtime_error("zero norm VJP");
    for (int mode=0;mode<3;++mode) {
      Scoring bad;
      if(mode==0)bad.read_device="model";
      if(mode==1)bad.control_device="model";
      if(mode==2){bad=fp32;bad.read_device="model";}
      bool rejected=false;
      try {bad.validate(at::Device(c10::DeviceType::PrivateUse1,0),mode!=2);}
      catch(const std::invalid_argument&) {rejected=true;}
      if(!rejected)throw std::runtime_error("unsupported scoring configuration accepted");
    }
    std::cout << "analytic FP32 Read values/VJPs/zero and unsupported configurations: passed\n";
    return 0;
  } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
