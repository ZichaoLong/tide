// 0770593 architecture §1/2: independent per-atom formula/autograd anchor.
#include "weighted_aggregate.h"
#include "portable_torch/runtime.hpp"
#include <torch/csrc/autograd/autograd.h>
#include <ATen/Parallel.h>
#include <cmath>
#include <iostream>
namespace {
using namespace tide::device_online;using Tensor=at::Tensor;
void require(bool yes,const char* message){if(!yes)throw std::runtime_error(message);}
void close(const Tensor& actual,const Tensor& expected){
  require(actual.sizes()==expected.sizes()&&actual.scalar_type()==expected.scalar_type(),"Aggregate comparison metadata");
  if(!at::allclose(actual.cpu(),expected,1e-5,1e-6)){
    std::cerr<<"actual="<<actual.cpu()<<" expected="<<expected<<'\n';throw std::runtime_error("resident Aggregate numeric difference");}
}
Tensor values(at::IntArrayRef shape,float offset){int64_t n=1;for(auto d:shape)n*=d;return (at::arange(n,at::kFloat)*.13+offset).sin().reshape(shape)*.2;}
std::vector<Tensor> tensors(const WeightedSourceParameters& w){return {w.a,w.b,w.embedding};}
struct Fixture {
  Tensor payload,owners,lengths,gradient,roots;
  WeightedSourceParameters weights;
  const std::vector<int64_t> hidden{4,2,3,1};
  explicit Fixture(int variant){
    payload=values({6,4,4},.2+variant*.3);
    payload.select(2,0).copy_(at::tensor({.3f,.4f,0.f,-.2f}).repeat({6,1}));
    payload[1].select(1,0).copy_(at::tensor({0.f,-.1f,0.f,0.f}));payload[3][0][0]=.5;
    owners=at::tensor(std::vector<int64_t>{0,0,1,1},at::kLong).repeat({6,1});owners[1].fill_(2);owners[4].fill_(3);
    owners[5].copy_(at::tensor(std::vector<int64_t>{1,0,1,0},at::kLong));
    lengths=at::tensor(variant?std::vector<int64_t>{2,3,0,1,3,4}:std::vector<int64_t>{4,2,0,1,2,3},at::kLong);
    gradient=values({6,4},.7);if(!variant)gradient[0].zero_();gradient[4].fill_(NAN);
    roots=at::tensor({1,1,1,1,0,1},at::kInt).to(at::kBool);
    weights={values({4,4,3},.1),values({4,3,4},.3),values({4,4},.5)};
    for(int bank=0;bank<4;++bank){weights.a[bank].slice(0,hidden[bank]).zero_();
      weights.b[bank].slice(1,hidden[bank]).zero_();weights.embedding[bank].slice(0,hidden[bank]).zero_();}
    for(int row=0;row<6;++row){auto n=lengths[row].item<int64_t>();payload[row].slice(0,n).fill_(NAN);owners[row].slice(0,n).fill_(-42);}
  }
};
void compare(const Fixture& f,float alpha,const WeightedAggregateTape& tape,const WeightedAggregateGradient& g){
  at::AutoGradMode enable(true);std::vector<Tensor> leaves,outputs,bars,expected;
  for(auto row:f.payload.unbind())for(auto atom:row.unbind())leaves.push_back(atom.clone().set_requires_grad(true));
  for(auto bank:tensors(f.weights))for(auto owner:bank.unbind())leaves.push_back(owner.clone().set_requires_grad(true));
  for(int row=0;row<6;++row){auto n=f.lengths[row].item<int64_t>();
    if(!n){expected.push_back(at::zeros({4},at::kFloat));continue;}
    std::vector<Tensor> masses;for(int j=0;j<n;++j)masses.push_back(leaves[row*4+j][0].clamp_min(0));
    auto total=at::stack(masses).sum(),mean=at::zeros({3},at::kFloat);
    if(total.item<float>()>0){std::vector<Tensor> terms;
      for(int j=0;j<n;++j){auto z=leaves[row*4+j].slice(0,1);auto transformed=z;
        if(alpha>0){auto owner=f.owners[row][j].item<int64_t>(),h=f.hidden[owner];
          auto a=leaves[24+owner].slice(0,0,h),b=leaves[28+owner].slice(1,0,h),e=leaves[32+owner].slice(0,0,h);
          auto normalized=z/at::sqrt(z.square().mean()+1e-5);
          transformed=z+alpha*at::mv(b,at::silu(at::mv(a,normalized)+e));}
        terms.push_back(masses[j]*transformed);}
      mean=at::stack(terms).sum(0)/total;
    }
    auto result=at::cat({total.reshape({1}),mean});expected.push_back(result.detach());
    if(f.roots[row].item<bool>()){outputs.push_back(result);bars.push_back(f.gradient[row]);}
  }
  close(tape.output,at::stack(expected));auto grads=torch::autograd::grad(outputs,leaves,bars,false,false,true);
  auto flags=g.connected.cpu();
  for(int row=0;row<6;++row)for(int j=0;j<4;++j){const auto& value=grads[row*4+j];
    require(flags[row][j].item<bool>()==value.defined(),"Aggregate atom None/zero differs");
    if(value.defined())try{close(g.payload[row][j],value);}catch(...){std::cerr<<"row="<<row<<" atom="<<j<<'\n';throw;}
    else require(g.payload[row][j].cpu().eq(0).all().item<bool>(),"absent Aggregate gradient");}
  for(int role=0;role<3;++role)for(int owner=0;owner<4;++owner){const auto& value=grads[24+role*4+owner];
    if(!g.parameters){require(!value.defined(),"alpha0 source parameter connected");continue;}
    require(g.owner_connected[owner].cpu().item<bool>()==value.defined(),"source owner None/zero differs");
    auto actual=tensors(*g.parameters)[role][owner];if(value.defined())close(actual,value);
    else require(actual.cpu().eq(0).all().item<bool>(),"absent source parameter gradient");}
}
void check(at::Device device,float alpha){
  at::NoGradGuard no_grad;Fixture initial(0);auto move=[&](const Tensor& t){return t.to(device);};
  auto payload=move(initial.payload),lengths=move(initial.lengths),owners=move(initial.owners);
  auto gradient=move(initial.gradient),roots=move(initial.roots);auto error=at::zeros({1},lengths.options().dtype(at::kInt));
  WeightedSourceParameters weights{move(initial.weights.a),move(initial.weights.b),move(initial.weights.embedding)};
  DeviceProgram p(device);p.limit_workspace(64*1024*1024);
  auto t=append_weighted_aggregate(p,payload,lengths,owners,weights,alpha,1e-5,error);
  auto g=append_weighted_aggregate_vjp(p,t,gradient,roots,error);
  auto committed=at::full_like(t.output,-7),ok=at::zeros({1},roots.options()),index=at::zeros_like(error),zero=at::zeros_like(error);
  auto publish=p.label(),done=p.label();p.equal(error,zero,ok);p.cast_index(ok,index);p.branch(index,{done,publish});
  p.mark(publish);p.copy(committed,t.output);p.mark(done);p.finish();
  for(int variant:{0,1}){Fixture f(variant);payload.copy_(f.payload);owners.copy_(f.owners);lengths.copy_(f.lengths);
    gradient.copy_(f.gradient);roots.copy_(f.roots);error.zero_();portable_torch::synchronize(device);p.run();
    require(error.cpu().item<int>()==0,"resident Aggregate refused valid fibers");
    try{compare(f,alpha,t,g);}catch(...){std::cerr<<"alpha="<<alpha<<" variant="<<variant<<'\n';throw;}
    close(committed,t.output.cpu());}
  DeviceProgram reverse(device);reverse.limit_workspace(64*1024*1024);
  auto saved=append_weighted_aggregate_vjp(reverse,t,gradient,roots,error);reverse.finish();
  payload.fill_(NAN);lengths.fill_(-42);owners.fill_(-42);for(auto w:tensors(weights))w.fill_(NAN);
  portable_torch::synchronize(device);reverse.run();require(error.cpu().item<int>()==0,"Aggregate reverse reread mutable sources");
  compare(Fixture(1),alpha,t,saved);reverse.close();
  auto before=committed.cpu();portable_torch::synchronize(device);p.run();
  require(error.cpu().item<int>()==2,"Aggregate accepted invalid dynamic length");close(committed,before);p.close();
  DeviceProgram refusal(device);bool refused=false;
  try{append_weighted_aggregate(refusal,payload,lengths,owners,weights,alpha,1e-5,error,1);}catch(const std::invalid_argument&){refused=true;}
  require(refused,"Aggregate scratch budget not enforced");refusal.close();
}
}
int main(int argc,char** argv){
  portable_torch::RuntimeSession runtime;
  try{auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("resident Aggregate requires explicit accelerator/FP32");
    auto device=portable_torch::resolve_device(args);at::set_num_threads(1);at::set_num_interop_threads(1);
    check(device,0);check(device,.25);
    std::cout<<"weighted-aggregate: passed configurations=2 physical_rows=24 logical_fibers=20 full_vjp=true duplicate_classes=true saved_versions=true\n";return 0;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}
}
