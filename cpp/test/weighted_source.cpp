// 0770593 architecture §2.2: independent scalar low-rank formula/Jacobian.
#include "tide/weighted.h"
#include "tide/aggregate.h"
#include <torch/csrc/autograd/autograd.h>
#include <cmath>
#include <stdexcept>

void check_weighted_source(const at::TensorOptions& o){
  using namespace tide;using namespace tide::weighted;
  auto close=[&](const Tensor& x,const Tensor& y){
    if(!at::allclose(x,y,o.dtype()==at::kDouble?1e-8:1e-5,o.dtype()==at::kDouble?1e-10:1e-6))
      throw std::runtime_error("source adaptation formula/derivative mismatch");
  };
  NodeWeights w{at::zeros({2},o),at::zeros({2,2},o),at::zeros({2},o),at::zeros({2},o)};
  const std::string p="aggregate_program.transforms.0.";
  std::vector<std::string> names{"a","b","embedding","a_left","a_right","b_left","b_right"};
  std::vector<double> values{.7,-.4,.2,.3,-.5,.6,.2};std::vector<Tensor> roots;
  for(size_t i=0;i<names.size();++i){
    auto x=at::full(names[i]=="embedding"?at::IntArrayRef{1}:at::IntArrayRef{1,1},values[i],o).set_requires_grad(true);
    w.extra[p+names[i]]=x;roots.push_back(x);
  }
  KVConfig c;c.width=c.summary_width=1;c.epsilon=1;
  configure_node(w,c,RegionalConfig{},SourceClasses{{},{0,0}},.4);
  w.aggregate_kernel->validate_weights(w,2);
  auto x=at::tensor({.3,.8},o).set_requires_grad(true);roots.push_back(x);
  AggregateInput r{0,2,{{0,Atom{0,0,0,0,0,0,x},at::ones({},o)},
                         {1,Atom{0,0,0,0,1,0,x},at::ones({},o)}}};
  const double a=.7-.15,b=-.4+.12,n=.8/std::sqrt(1+.8*.8),h=a*n+.2,s=1/(1+std::exp(-h));
  const double df=b*s*(1+h*(1-s)),da=.4*df*n,db=.4*h*s;
  std::vector<double> expected{da,db,.4*df,-.5*da,.3*da,.2*db,.6*db};
  for(Index mode=0;mode<3;++mode){
    auto y=(mode==0?w.aggregate_kernel->step(w,r):mode==1?w.aggregate_kernel->batch(w,{r})[0]:
            w.aggregate_kernel->batch_grad(w,{r})[0]).value;
    close(y,at::tensor({.6,.8+.4*b*h*s},o));
    auto grads=torch::autograd::grad({y[1]},roots,{},true);
    for(size_t i=0;i<expected.size();++i)close(grads[i],at::full_like(roots[i],expected[i]));
    close(grads.back(),at::tensor({0.,1+.4*df*a/std::pow(1+.8*.8,1.5)},o));
    close(torch::autograd::grad({y[0]},{x})[0],at::tensor({2.,0.},o));
  }
  auto bad=w;bad.extra.erase(p+"b_right");bool rejected=false;
  try{bad.aggregate_kernel->validate_weights(bad,2);}catch(const std::exception&){rejected=true;}
  if(!rejected)throw std::runtime_error("source adaptation accepted partial factors");
  bad=w;bad.extra[p+"a_left"]=at::zeros({1,2},o);rejected=false;
  try{bad.aggregate_kernel->validate_weights(bad,2);}catch(const std::exception&){rejected=true;}
  if(!rejected)throw std::runtime_error("source adaptation accepted incompatible factors");
}
