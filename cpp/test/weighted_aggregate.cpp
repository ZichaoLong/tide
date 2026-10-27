// Standalone architecture §2: literal multiple-atoms-per-class batch witness.
#include "tide/aggregate.h"
#include "tide/weighted.h"
#include <torch/csrc/autograd/autograd.h>
#include <ATen/core/grad_mode.h>
#include <cmath>
#include <stdexcept>

void check_weighted_aggregate(const at::TensorOptions& o){
  using namespace tide;using namespace tide::weighted;
  NodeWeights w{at::zeros({2},o),at::zeros({2,2},o),at::zeros({2},o),at::zeros({2},o)};
  auto a=at::full({1,1},.7,o).set_requires_grad(true),b=at::full({1,1},-.4,o).set_requires_grad(true);
  for(Index label=0;label<2;++label){auto p="aggregate_program.transforms."+std::to_string(label);
    w.extra[p+".a"]=a;w.extra[p+".b"]=b;w.extra[p+".embedding"]=at::full({1},label?-.3:.2,o).set_requires_grad(true);
  }
  KVConfig c;c.width=c.summary_width=1;c.epsilon=1;
  configure_node(w,c,RegionalConfig{},SourceClasses{{},{0,0,1,1}},.5);
  w.aggregate_kernel->validate_weights(w,4);
  auto x=at::tensor({.2,2.},o).set_requires_grad(true),y=at::tensor({.5,-.4},o).set_requires_grad(true);
  auto source=[&](Index sample,Index port,const Tensor& value){
    return SourceInput{port,Atom{sample,0,sample,0,port,0,value},at::ones({},o)};
  };
  std::vector<AggregateInput> requests{
    {0,4,{source(0,0,x),source(0,1,x),source(0,2,y)}},
    {1,4,{source(1,0,at::tensor({0.,9.},o).set_requires_grad(true)),source(1,3,at::tensor({-.2,-7.},o).set_requires_grad(true))}},
    {2,4,{source(2,2,y)}}};
  auto actual=w.aggregate_kernel->batch(w,requests);
  auto transform=[](double value,double embedding){
    double h=.7*value/std::sqrt(1+value*value)+embedding;
    return value-.2*h/(1+std::exp(-h));
  };
  std::vector<Tensor> expected{at::tensor({.9,(.4*transform(2.,.2)+.5*transform(-.4,-.3))/.9},o),
    at::zeros({2},o),at::tensor({.5,transform(-.4,-.3)},o)};
  for(size_t i=0;i<expected.size();++i)
    if(!at::allclose(expected[i],actual[i].value,expected[i].scalar_type()==at::kDouble?1e-8:1e-5,expected[i].scalar_type()==at::kDouble?1e-10:1e-6)
       ||!actual[i].contributions.empty())throw std::runtime_error("weighted Aggregate packed literal mismatch");
  auto gradient=torch::autograd::grad({actual[0].value[0]},{x})[0];
  if(!at::equal(gradient,at::tensor({2.,0.},o))||!w.aggregate_kernel->joint_batch())
    throw std::runtime_error("weighted Aggregate lost repeated class atom multiplicity");
  auto batched=w.aggregate_kernel->batch_grad(w,requests);
  if(!w.aggregate_kernel->batched_autograd())throw std::runtime_error("missing Aggregate VJP capability");
  std::vector<Tensor> roots{x,y,a,b,w.extra.at("aggregate_program.transforms.0.embedding"),
    w.extra.at("aggregate_program.transforms.1.embedding"),requests[1].sources[0].atom.value,requests[1].sources[1].atom.value};
  for(size_t i=0;i<requests.size();++i){
    auto scalar=w.aggregate_kernel->step(w,requests[i]).value;
    for(bool zero:{false,true}){
      auto loss=[&](const Tensor& t){return t.square().sum()*(zero?0.:.7);};
      auto ref=torch::autograd::grad({loss(scalar)},roots,{},true,false,true);
      auto got=torch::autograd::grad({loss(batched[i].value)},roots,{},true,false,true);
      for(size_t j=0;j<roots.size();++j)
        if(ref[j].defined()!=got[j].defined()||(ref[j].defined()&&!at::allclose(ref[j],got[j],
            o.dtype()==at::kDouble?1e-8:1e-5,o.dtype()==at::kDouble?1e-10:1e-6)))
          throw std::runtime_error("weighted Aggregate batch VJP changed value or connectivity");
    }
  }
  bool high=false,version=false;
  try{torch::autograd::grad({batched[0].value.sum()},{x},{},true,true);}
  catch(const std::exception& e){high=std::string(e.what()).find("first-order")!=std::string::npos;}
  {at::NoGradGuard guard;a.add_(.1);}
  try{torch::autograd::grad({batched[0].value.sum()},{x});}
  catch(const std::exception& e){version=std::string(e.what()).find("modified by an inplace operation")!=std::string::npos;}
  if(!high||!version)throw std::runtime_error("weighted Aggregate VJP did not reject version or higher-order request");
}
