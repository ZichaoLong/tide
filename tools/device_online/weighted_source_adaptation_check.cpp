// 0770593 architecture §2.2: independent low-rank source formulas + autograd.
#include "weighted_source_adaptation.h"
#include "portable_torch/runtime.hpp"
#include <torch/csrc/autograd/autograd.h>
#include <ATen/Parallel.h>
#include <cmath>
#include <iostream>
namespace {
using namespace tide::device_online;using Tensor=at::Tensor;
void require(bool b,const char* text){if(!b)throw std::runtime_error(text);}
void close(const Tensor& a,const Tensor& b){
  require(a.sizes()==b.sizes()&&a.scalar_type()==b.scalar_type(),"adaptation metadata differs");
  if(!at::allclose(a.cpu(),b,1e-5,1e-6)){std::cerr<<"actual="<<a.cpu()<<" expected="<<b<<'\n';throw std::runtime_error("source adaptation numeric difference");}
}
Tensor values(at::IntArrayRef shape,float offset){int64_t n=1;for(auto d:shape)n*=d;return (at::arange(n,at::kFloat)*.13+offset).sin().reshape(shape)*.2;}
std::vector<Tensor> tensors(const WeightedSourceParameters& p){return {p.a,p.b,p.embedding};}
std::vector<Tensor> tensors(const WeightedSourceFactors& f){return {f.a_left,f.a_right,f.b_left,f.b_right};}
struct Fixture {
  WeightedSourceParameters base;
  WeightedSourceFactors factors;
  Tensor payload,lengths,owners,gradient,roots;
  explicit Fixture(int variant){
    base={values({3,4,3},.1),values({3,3,4},.2),values({3,4},.3)};
    factors={values({3,4,2},.4),values({3,2,3},.5),values({3,3,2},.6),values({3,2,4},.7)};
    for(auto x:tensors(factors))x+=variant*.02;
    payload=values({4,3,4},.8+variant*.1);payload.select(2,0).copy_(at::tensor({.2f,.4f,0.f}).repeat({4,1}));
    payload[1].select(1,0).zero_();
    lengths=at::tensor(variant?std::vector<int64_t>{2,3,1,3}:std::vector<int64_t>{3,1,2,3},at::kLong);
    owners=at::tensor(std::vector<int64_t>{0,1,0},at::kLong).repeat({4,1});owners[1].fill_(2);owners[3].fill_(2);
    roots=at::tensor({1,1,1,0},at::kInt).to(at::kBool);gradient=values({4,4},.9);
    gradient[3].fill_(NAN);if(!variant)gradient[0].zero_();
    for(int row=0;row<4;++row){auto n=lengths[row].item<int64_t>();payload[row].slice(0,n).fill_(NAN);owners[row].slice(0,n).fill_(-42);}
  }
};
void compare(const Fixture& f,const WeightedAggregateTape& t,const WeightedAggregateGradient& ag,
             const WeightedSourceAdaptation& adapted,const WeightedSourceAdaptationGradient& sg){
  at::AutoGradMode enable(true);std::vector<Tensor> leaves;
  for(auto x:tensors(f.base))for(auto v:x.unbind())leaves.push_back(v.clone().set_requires_grad(true));
  for(auto x:tensors(f.factors))for(auto v:x.unbind())leaves.push_back(v.clone().set_requires_grad(true));
  for(auto row:f.payload.unbind())for(auto v:row.unbind())leaves.push_back(v.clone().set_requires_grad(true));
  std::vector<Tensor> a,b,expected,ys,bars;
  for(int bank=0;bank<3;++bank){a.push_back(leaves[bank]+at::mm(leaves[9+bank],leaves[12+bank]));
    b.push_back(leaves[3+bank]+at::mm(leaves[15+bank],leaves[18+bank]));}
  close(adapted.effective.a,at::stack(a));close(adapted.effective.b,at::stack(b));
  for(int row=0;row<4;++row){const auto n=f.lengths[row].item<int64_t>();std::vector<Tensor> mass,terms;
    for(int j=0;j<n;++j)mass.push_back(leaves[21+row*3+j][0].clamp_min(0));
    auto total=at::stack(mass).sum(),mean=at::zeros({3},at::kFloat);
    if(total.item<float>()>0){for(int j=0;j<n;++j){auto z=leaves[21+row*3+j].slice(0,1);auto bank=f.owners[row][j].item<int64_t>();
      auto normalized=z/at::sqrt(1e-5+z.square().mean());auto source=at::mv(b[bank],at::silu(at::mv(a[bank],normalized)+leaves[6+bank]));
      terms.push_back(mass[j]*(z+.25*source));}mean=at::stack(terms).sum(0)/total;}
    auto out=at::cat({total.reshape({1}),mean});expected.push_back(out.detach());
    if(f.roots[row].item<bool>()){ys.push_back(out);bars.push_back(f.gradient[row]);}}
  close(t.output,at::stack(expected));auto grads=torch::autograd::grad(ys,leaves,bars,false,false,true);
  auto roles=tensors(sg.base);auto extra=tensors(sg.factors);roles.insert(roles.end(),extra.begin(),extra.end());
  for(int role=0;role<7;++role)for(int bank=0;bank<3;++bank){const auto& expected=grads[role*3+bank];
    require(sg.connected[bank].cpu().item<bool>()==expected.defined(),"adaptation None/zero differs");
    if(expected.defined())close(roles[role][bank],expected);
    else require(roles[role][bank].cpu().eq(0).all().item<bool>(),"absent adaptation gradient");}
  for(int row=0;row<4;++row)for(int j=0;j<3;++j){const auto& expected=grads[21+row*3+j];
    require(ag.connected[row][j].cpu().item<bool>()==expected.defined(),"adapted atom None/zero differs");
    if(expected.defined())close(ag.payload[row][j],expected);}
}
void check(at::Device device){
  at::NoGradGuard no_grad;Fixture initial(0);auto move=[&](const Tensor& t){return t.to(device);};
  WeightedSourceParameters base{move(initial.base.a),move(initial.base.b),move(initial.base.embedding)};
  WeightedSourceFactors factors{move(initial.factors.a_left),move(initial.factors.a_right),move(initial.factors.b_left),move(initial.factors.b_right)};
  auto payload=move(initial.payload),lengths=move(initial.lengths),owners=move(initial.owners),bar=move(initial.gradient),roots=move(initial.roots);
  auto error=at::zeros({1},lengths.options().dtype(at::kInt));DeviceProgram p(device);p.limit_workspace(64*1024*1024);
  auto adapted=append_weighted_source_adaptation(p,base,factors,error);
  auto aggregate=append_weighted_aggregate(p,payload,lengths,owners,adapted.effective,.25,1e-5,error);
  auto ag=append_weighted_aggregate_vjp(p,aggregate,bar,roots,error);
  auto sg=append_weighted_source_adaptation_vjp(p,adapted,*ag.parameters,ag.owner_connected,error);
  auto committed=at::full_like(aggregate.output,-7),ok=at::zeros({1},roots.options()),index=at::zeros_like(error),zero=at::zeros_like(error);
  auto publish=p.label(),done=p.label();p.equal(error,zero,ok);p.cast_index(ok,index);p.branch(index,{done,publish});
  p.mark(publish);p.copy(committed,aggregate.output);p.mark(done);p.finish();
  for(int variant:{0,1}){Fixture f(variant);auto src=tensors(f.factors),dst=tensors(factors);
    for(int i=0;i<4;++i)dst[i].copy_(src[i]);payload.copy_(f.payload);lengths.copy_(f.lengths);owners.copy_(f.owners);bar.copy_(f.gradient);
    portable_torch::synchronize(device);p.run();require(error.cpu().item<int>()==0,"source adaptation refused valid composition");compare(f,aggregate,ag,adapted,sg);}
  DeviceProgram reverse(device);reverse.limit_workspace(64*1024*1024);
  auto rg=append_weighted_aggregate_vjp(reverse,aggregate,bar,roots,error);
  auto rs=append_weighted_source_adaptation_vjp(reverse,adapted,*rg.parameters,rg.owner_connected,error);reverse.finish();
  for(auto x:tensors(base))x.fill_(NAN);for(auto x:tensors(factors))x.fill_(NAN);payload.fill_(NAN);lengths.fill_(-42);owners.fill_(-42);
  portable_torch::synchronize(device);reverse.run();require(error.cpu().item<int>()==0,"source reverse reread mutated factors");
  compare(Fixture(1),aggregate,rg,adapted,rs);reverse.close();
  auto before=committed.cpu();p.run();require(error.cpu().item<int>()==20,"source nonfinite input not rejected");
  close(committed,before);p.close();
  DeviceProgram refusal(device);bool rejected=false;
  try{append_weighted_source_adaptation(refusal,base,factors,error,1);}catch(const std::invalid_argument&){rejected=true;}
  require(rejected,"source adaptation omitted scratch budget");refusal.close();
}
}
int main(int argc,char** argv){portable_torch::RuntimeSession runtime;
  try{auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("source adaptation requires explicit accelerator/FP32");
    auto device=portable_torch::resolve_device(args);at::set_num_threads(1);at::set_num_interop_threads(1);check(device);
    std::cout<<"weighted-source: passed banks=3 factor_roles=4 full_aggregate_vjp=true replay=true saved_versions=true failed_window_no_commit=true\n";return 0;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}}
