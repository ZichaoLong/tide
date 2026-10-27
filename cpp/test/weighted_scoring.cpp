// Standalone literal equation (2–5) witnesses, separate from Python tests.
#include "../src/weighted_scoring.h"
#include <torch/csrc/autograd/autograd.h>
#include <cmath>
#include <stdexcept>

namespace {
using namespace tide;
void close(const Tensor& a,const Tensor& b){
  if(!at::allclose(a,b,a.scalar_type()==at::kDouble?1e-8:1e-5,a.scalar_type()==at::kDouble?1e-10:1e-6))
    throw std::runtime_error("standalone weighted scoring mismatch");
}
}
void check_weighted_scoring(const at::TensorOptions& o){
  using namespace tide;using namespace tide::weighted;using namespace tide::weighted::detail;
  RegionalConfig r;r.local_score="match";r.region_evidence="grouped";r.selection_width=1;
  r.hidden_width=1;r.region_control_width=1;r.alpha_match=1.5;r.alpha_res=.25;r.validate();
  NodeWeights w;w.bias=at::zeros({2},o);
  const std::array<std::string,3> names{"cur","mem","stat"};const std::array<double,3> biases{1.,2.,-.5};
  for(const auto& family:{"score","evidence"})for(size_t i=0;i<3;++i){
    auto p=std::string("read_program.")+family+".encoders."+names[i];
    w.extra[p+".in_weight"]=at::zeros({1,static_cast<Index>(i+1)},o);w.extra[p+".hidden_bias"]=at::zeros({1},o);
    w.extra[p+".out_weight"]=at::zeros({1,1},o);w.extra[p+".out_bias"]=at::full({1},biases[i],o).set_requires_grad(true);
  }
  for(const auto& name:{"gate","up"})w.extra[std::string("read_program.score.")+name]=at::zeros({1,3},o);
  w.extra["read_program.score.gate_bias"]=at::full({1},std::log(2.),o);w.extra["read_program.score.up_bias"]=at::full({1},3.,o);
  w.extra["read_program.score.vector"]=at::full({1},2.,o);w.extra["read_program.score.bias"]=at::full({},.2,o);
  w.extra["read_program.evidence.cur"]=at::full({1,1},2.,o);w.extra["read_program.evidence.mem"]=at::full({1,1},-1.,o);
  w.extra["read_program.evidence.stat"]=at::full({1,1},.5,o);w.extra["read_program.evidence.interaction"]=at::full({1,1},3.,o);
  w.extra["read_program.evidence.bias"]=at::full({1},.25,o);
  Groups g{at::tensor({.3},o),at::tensor({.2,-.1},o),at::tensor({.4,.5,.6},o)};auto f=at::zeros({8},o);
  auto s=local_score(w,r,f,g,1.);const double x=1/std::sqrt(2.),e=2/std::sqrt(5.),z=-1/std::sqrt(5.);
  close(s,at::full({},.2+1.5*x*e+std::log(2.),o));
  auto grad=torch::autograd::grad({s},{w.extra.at("read_program.score.encoders.cur.out_bias")})[0];
  close(grad,at::full({1},1.5*e/std::pow(2.,1.5),o));
  double value=2*x-e+.5*z+3*x*e+.25;
  close(region_evidence(w,r,f,g,1.),at::full({1},value/std::sqrt(1+value*value),o));
  r.region_score="att";r.region_key="content_stats";r.query_weight="received";r.alpha_loc=.5;r.alpha_sel=1.3;r.validate();
  w.extra["read_program.key_weight"]=at::ones({1,4},o);w.extra["read_program.key_bias"]=at::full({1},.1,o);
  close(memory_key(w,r,g,1.),at::full({1},1.3,o));
  RegionWeights region;region.extra={{"query.in_weight",at::ones({1,1},o)},{"query.out_weight",at::ones({1,1},o)},
    {"query.hidden_bias",at::zeros({1},o)},{"query.out_bias",at::zeros({1},o)}};
  std::map<std::string,Tensor> fields{{"w",at::tensor({.2,.3},o)},{"score",at::tensor({.3,.2},o)},
    {"x",at::tensor({1.,2.},o).reshape({2,1})},{"key",at::tensor({1.,-1.},o).reshape({2,1})}};
  const double q=.8/(1+std::exp(-.8));
  close(region_scores(region,r,fields),at::tensor({.15+1.3*q,.1-1.3*q},o));
}
