// Independent standalone literal examples: affine-delta §3.3, GDN/SSD §3.4.
#include "tide/weighted.h"
#include "tide/weighted_convolution.h"
#include "tide/full.h"
#include <torch/csrc/autograd/autograd.h>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {
using namespace tide;
void close(const Tensor& x,const Tensor& y){
  if(!at::allclose(x,y,x.scalar_type()==at::kDouble?1e-8:1e-5,x.scalar_type()==at::kDouble?1e-10:1e-6))
    throw std::runtime_error("weighted recurrent literal formula mismatch");
}
double inverse_silu(double y){
  double lo=0,hi=std::max(2.,2*y);
  for(int i=0;i<80;++i){auto x=(lo+hi)/2;if(x/(1+std::exp(-x))<y)lo=x;else hi=x;}
  return (lo+hi)/2;
}
}
void check_weighted_recurrent(const at::TensorOptions& o){
  using namespace tide;using namespace tide::weighted;
  for(const std::string family:{"affine","delta","gated_delta","ssd"}){
    RecurrentConfig c;c.memory_family=family;c.width=c.summary_width=c.key_width=c.value_width=c.ffn_width=1;
    c.epsilon=1;c.alpha_ff=0;c.memory_read="state";
    if(!c.basic())c.output_epsilon=1;
    if(family=="gated_delta")c.qk_epsilon=1;
    NodeWeights w{at::zeros({2},o),at::zeros({2,2},o),at::zeros({2},o),at::zeros({2},o)};
    auto& p=w.extra;p["kernel.summary"]=at::full({1,1},std::sqrt(2.),o);
    for(const std::string name:{"gate","up","down"})p["full_program.content."+name]=at::zeros({1,1},o);
    if(c.basic()){
      p["kernel.key"]=at::full({1,1},std::sqrt(2.),o);p["kernel.value"]=at::full({1,1},std::sqrt(2.),o).set_requires_grad(true);
      p["kernel.retain"]=at::zeros({1},o);p["kernel.retain_bias"]=at::zeros({},o);
      if(family=="delta"){p["kernel.write"]=at::zeros({1},o);p["kernel.write_bias"]=at::zeros({},o);}
      p["kernel.read_query"]=at::full({1,1},std::sqrt(2.),o);p["kernel.read_out"]=at::ones({2,1,1},o);
      p["full_program.content.query"]=at::full({1,1},std::sqrt(2.),o);
      p["full_program.content.out"]=at::tensor({0.,1.},o).reshape({2,1,1});
    }else {
      p["kernel.conv.weight"]=at::ones({1,3},o);p["kernel.conv.bias"]=at::zeros({3},o);
      p["kernel.read_out"]=at::ones({1,1},o);p["kernel.log_rate"]=at::zeros({1},o);
      p["kernel.delta_bias"]=at::zeros({1},o);p["full_program.content.out"]=at::ones({1,1},o);
      if(family=="gated_delta"){
        auto a=inverse_silu(1/std::sqrt(3.));
        p["kernel.projection"]=std::sqrt(2.)*at::tensor({a,a,inverse_silu(2)},o).reshape({3,1});
        p["kernel.delta"]=at::zeros({1,1},o);p["kernel.write"]=at::zeros({1,1},o);p["kernel.write_bias"]=at::zeros({1},o);
        p["full_program.content.rho"]=at::ones({1,1},o);p["full_program.content.output_gate"]=at::full({1,1},std::sqrt(2.),o);
      }else {
        p["kernel.projection"]=std::sqrt(2.)*at::tensor({1.,inverse_silu(2),inverse_silu(1),inverse_silu(3),0.},o).reshape({5,1});
        p["kernel.log_rate"]=at::full({1},std::log(2*std::log(2.)),o);
        p["kernel.delta_bias"]=at::full({1},std::log(std::expm1(.5)),o);
        p["full_program.content.direct"]=at::full({1},.25,o);p["full_program.content.rho"]=at::ones({1},o);
      }
    }
    RegionalConfig r;configure_node(w,c,r,SourceClasses{{},{0}},0);
    w.kernel->validate_weights(w);w.full_kernel->validate_weights(w,1);
    auto initial=w.kernel->initial(w);const auto receive=ContentView{at::ones({2},o)};
    auto next=w.kernel->step(w,initial,receive,0);w.kernel->validate_state(w,next);
    auto expected_h=family=="delta"||family=="gated_delta"?.5:1.;
    close(next.slots.at("H"),at::full_like(next.slots.at("H"),expected_h));
    for(const auto& name:{"Z","W"})close(next.slots.at(name),at::ones_like(next.slots.at(name)));
    auto result=content(w,next,0,at::ones({1},o),c);
    double expected=1+expected_h;
    if(family=="gated_delta")expected=1+1/(1+std::exp(-1.))/std::sqrt(17.);
    if(family=="ssd"){auto y=3.5/(1+std::exp(-1.));expected=1+y/std::sqrt(1+y*y);}
    close(result,at::full({1},expected,o));
    if(c.basic()){
      auto grad=torch::autograd::grad({result.sum()},{p.at("kernel.value")})[0];
      close(grad,at::full({1,1},expected_h/std::sqrt(2.),o));
    }else {
      auto second=w.kernel->step(w,next,receive,1);
      close(second.slots.at("H"),at::full_like(second.slots.at("H"),family=="gated_delta"?23./32:1.5));
    }
    auto zero=w.kernel->step(w,next,ContentView{at::tensor({0.,99.},o)},7);
    if(zero.last_time!=next.last_time||!at::equal(zero.slots.at("time"),next.slots.at("time")))
      throw std::runtime_error("zero receive changed recurrent raw timestamp");
    const auto clock=(Index{1}<<54)+3;
    auto later=w.kernel->step(w,next,receive,clock);
    if(later.slots.at("time").item<Index>()!=clock)throw std::runtime_error("recurrent timestamp lost precision");
  }
  // Fixed slots, weak-event continuity and exact endpoint at INT64_MAX.
  ConvCache b{at::tensor({0,2},o.dtype(at::kLong)),at::tensor({4.,1.},o).reshape({2,1})};
  auto weight=at::tensor({1.,2.,3.},o).reshape({3,1});
  close(convolve(b,4,at::full({1},6.,o),weight,at::zeros({1},o),2),at::full({1},20.,o));
  auto after=conv_write(b,4,at::full({},.2,o),at::full({1},6.,o),3,2);
  close(convolve(after,4,at::full({1},6.,o),weight,at::zeros({1},o),2),at::full({1},20.,o));
  auto projected=conv_effective(b,std::numeric_limits<Index>::max(),3,2);
  if(projected.first[-1].item<Index>()!=std::numeric_limits<Index>::max())throw std::runtime_error("projection overflow");
}
