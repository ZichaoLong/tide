// Independent scalar RegionKernel + autograd against the composed resident step.
#include "weighted_region.h"
#include "portable_torch/runtime.hpp"
#include <torch/csrc/autograd/autograd.h>
#include <ATen/Parallel.h>
#include <cmath>
#include <iostream>
namespace {
using namespace tide;using namespace tide::device_online;
void require(bool b,const char* text){if(!b)throw std::runtime_error(text);}
void close(const Tensor& a,const Tensor& b){
  require(a.sizes()==b.sizes()&&a.scalar_type()==b.scalar_type(),"region comparison metadata");
  if(!at::allclose(a.cpu(),b,1e-5,1e-6)){
    std::cerr<<"actual="<<a.cpu()<<" expected="<<b<<'\n';throw std::runtime_error("resident region numeric difference");}
}
std::vector<Tensor> tensors(const WeightedRegionFields& f){return {f.received,f.caps,f.score,f.evidence,f.current,f.key};}
std::vector<Tensor> tensors(const WeightedHeadParameters& p){return {p.matrix,p.hidden_bias,p.projection,p.bias};}
Tensor values(at::IntArrayRef shape,float shift){int64_t n=1;for(auto d:shape)n*=d;return (at::arange(n,at::kFloat)*.13+shift).sin().reshape(shape)*.2;}
WeightedHeadParameters head(int64_t in,int64_t hidden,int64_t out){
  return {values({3,hidden,in},.1),values({3,hidden},.2),values({3,out,hidden},.3),values({3,out},.4)};
}
struct Fixture {
  WeightedRegionFields fields;
  WeightedRegionParameters weights;
  Tensor ids,times,lengths,roots,gradient;
  explicit Fixture(int variant){
    fields={at::tensor({.2f,.1f,.3f,0.f}).repeat({5,1}),at::tensor({.3f,.5f,.6f,0.f}).repeat({5,1}),
      at::tensor({.1f,-.2f,.8f,0.f}).repeat({5,1}),values({5,4,2},.4),values({5,4,2},.7),values({5,4,2},.9)};
    fields.received[1].zero_();fields.caps[1].zero_();fields.score+=variant*.03;
    weights.context=head(4,3,1);weights.query=head(2,2,2);
    weights.context_owners=at::tensor(std::vector<int64_t>{0,1,42,0,1},at::kLong);
    weights.query_owners=at::tensor(std::vector<int64_t>{1,0,42,1,0},at::kLong);
    lengths=at::tensor(variant?std::vector<int64_t>{2,3,0,3,1}:std::vector<int64_t>{3,2,0,1,3},at::kLong);
    ids=at::tensor(std::vector<int64_t>{(1LL<<60)+7,(1LL<<60)+2,(1LL<<60)+4,-1},at::kLong).repeat({5,1});
    times=at::arange(5,at::kLong)+(1LL<<60)+variant*200;times[2]=-42;
    roots=at::tensor({1,1,1,1,0},at::kInt).to(at::kBool);
    gradient=at::tensor({.3f,-.7f,.9f,.1f}).repeat({5,1});if(!variant)gradient[0].zero_();
    for(int64_t row=0;row<5;++row){const auto n=lengths[row].item<int64_t>();
      for(auto x:tensors(fields))x[row].slice(0,n).fill_(NAN);
      ids[row].slice(0,n).fill_(-1);gradient[row].slice(0,n).fill_(NAN);}
    gradient[4].fill_(NAN);
  }
};
void compare(const Fixture& f,const weighted::RegionalConfig& c,const WeightedRegionTape& t,const WeightedRegionGradient& g){
  at::AutoGradMode mode(true);std::vector<Tensor> leaves,outputs,bars,expected_rows;
  // Each field row and each parameter owner is an independent mathematical root.
  auto input=tensors(f.fields);std::vector<std::vector<Tensor>> row_fields(5);
  for(int64_t row=0;row<5;++row){auto n=f.lengths[row].item<int64_t>();
    for(auto x:input){auto v=x[row].slice(0,0,n).clone().set_requires_grad(true);row_fields[row].push_back(v);leaves.push_back(v);}}
  std::vector<Tensor> params;
  for(auto bank:{f.weights.context,f.weights.query})for(auto x:tensors(bank))for(auto v:x.unbind())params.push_back(v.clone().set_requires_grad(true));
  leaves.insert(leaves.end(),params.begin(),params.end());auto active=t.active.cpu(),history=t.history_times.cpu(),present=t.history_present.cpu();
  for(int64_t row=0;row<5;++row){
    auto n=f.lengths[row].item<int64_t>();auto ids=f.ids[row];
    if(!n){expected_rows.push_back(at::zeros_like(f.gradient[row]));require(!present[row].item<bool>()&&history[row].item<int64_t>()==-1,"absent region history");continue;}
    RegionWeights w;
    if(c.region_control_mode=="context"){
      auto id=f.weights.context_owners[row].item<int64_t>();w.extra={{"head.matrix",params[id]},
        {"head.hidden_bias",params[3+id]},{"head.vector",params[6+id].squeeze(0)},{"head.bias",params[9+id].squeeze(0)}};}
    if(c.region_score=="att"){
      auto id=f.weights.query_owners[row].item<int64_t>();w.extra["query.in_weight"]=params[12+id];w.extra["query.hidden_bias"]=params[15+id];
      w.extra["query.out_weight"]=params[18+id];w.extra["query.out_bias"]=params[21+id];}
    weighted::configure_region(w,c);std::vector<Candidate> candidates;
    const std::vector<Index> members{(1LL<<60)+7,(1LL<<60)+2,(1LL<<60)+4,(1LL<<60)+8};
    for(int64_t j=0;j<n;++j){const auto& r=row_fields[row];
      TensorFields fields{{"w",r[0][j]},{"mu",r[1][j]},{"score",r[2][j]}};
      if(c.regional())fields["evidence"]=r[3][j];if(c.region_score=="att"){fields["x"]=r[4][j];fields["key"]=r[5][j];}
      candidates.push_back({ids[j].item<int64_t>(),{},fields});}
    Region spec{c.capacity(4)};RegionLayout layout{spec,members};History initial;
    w.kernel->validate_weights(w,layout);
    auto time=f.times[row].item<int64_t>();RegionInput request{initial,time,candidates,layout,f.fields.received.options()};
    auto result=w.kernel->step(w,request);std::vector<Tensor> values;
    for(int64_t j=0;j<n;++j){values.push_back(result.controls.at(members[j]));
      require(active[row][j].item<bool>()==(result.active.count(members[j])!=0),"region active support differs");}
    require(present[row].item<bool>()&&history[row].item<int64_t>()==result.history.last_time,"region int64 history differs");
    auto vector=at::stack(values);expected_rows.push_back(at::cat({vector.detach(),at::zeros({4-n},vector.options())}));
    if(f.roots[row].item<bool>()&&vector.requires_grad()){outputs.push_back(vector);bars.push_back(f.gradient[row].slice(0,0,n));}
  }
  close(t.controls,at::stack(expected_rows));auto expected=torch::autograd::grad(outputs,leaves,bars,false,false,true);
  auto gradients=tensors(g.fields);auto flags=g.connected.cpu();
  for(int64_t row=0;row<5;++row)for(int64_t field=0;field<6;++field){
    const auto n=f.lengths[row].item<int64_t>();const auto& value=expected[row*6+field];
    if(flags[row][field].item<bool>()!=value.defined()){
      std::cerr<<"row="<<row<<" field="<<field<<'\n';throw std::runtime_error("region field None/zero differs");}
    if(value.defined())try{close(gradients[field][row].slice(0,0,n),value);}
      catch(...){std::cerr<<"row="<<row<<" field="<<field<<'\n';throw;}
    else require(gradients[field][row].cpu().eq(0).all().item<bool>(),"absent region field gradient");
    require(gradients[field][row].slice(0,n).cpu().eq(0).all().item<bool>(),"region gradient padding");}
  for(int64_t kind=0;kind<2;++kind){const auto& head=kind?g.query_head:g.context_head;
    for(int64_t field=0;field<4;++field)for(int64_t bank=0;bank<3;++bank){
      const auto& value=expected[30+kind*12+field*3+bank];
      if(!head){require(!value.defined(),"missing region head gradient");continue;}
      require(head->owner_connected[bank].cpu().item<bool>()==value.defined(),"region head owner None/zero differs");
      auto actual=tensors(head->parameters)[field][bank];if(value.defined())close(actual,value);
      else require(actual.cpu().eq(0).all().item<bool>(),"absent region parameter gradient");}}
}
void check(at::Device device,const weighted::RegionalConfig& c){
  at::NoGradGuard no_grad;Fixture first(0);auto move=[&](Tensor x){return x.to(device);};
  WeightedRegionFields fields{move(first.fields.received),move(first.fields.caps),move(first.fields.score),
    move(first.fields.evidence),move(first.fields.current),move(first.fields.key)};
  auto move_head=[&](const WeightedHeadParameters& p){return WeightedHeadParameters{move(p.matrix),move(p.hidden_bias),move(p.projection),move(p.bias)};};
  WeightedRegionParameters weights{move_head(first.weights.context),move_head(first.weights.query),move(first.weights.context_owners),move(first.weights.query_owners)};
  auto ids=move(first.ids),times=move(first.times),lengths=move(first.lengths),roots=move(first.roots),gradient=move(first.gradient);
  auto error=at::zeros({1},ids.options().dtype(at::kInt));DeviceProgram p(device);p.limit_workspace(64*1024*1024);
  auto t=append_weighted_region(p,fields,ids,times,lengths,weights,c,error);auto g=append_weighted_region_vjp(p,t,gradient,roots,error);
  auto committed=at::full_like(t.controls,-7),ok=at::zeros({1},roots.options()),index=at::zeros_like(error),zero=at::zeros_like(error);
  auto publish=p.label(),done=p.label();p.equal(error,zero,ok);p.cast_index(ok,index);
  p.branch(index,{done,publish});p.mark(publish);p.copy(committed,t.controls);p.mark(done);p.finish();
  for(int variant:{0,1}){
    Fixture f(variant);auto src=tensors(f.fields),dst=tensors(fields);for(int j=0;j<6;++j)dst[j].copy_(src[j]);
    ids.copy_(f.ids);times.copy_(f.times);lengths.copy_(f.lengths);gradient.copy_(f.gradient);roots.copy_(f.roots);error.zero_();
    portable_torch::synchronize(device);p.run();
    require(error.cpu().item<int>()==0,"resident complete region refused valid rows");
    try{compare(f,c,t,g);}catch(...){std::cerr<<"region configuration="<<c.selection_mode<<" score="<<c.region_score<<" variant="<<variant<<'\n';throw;}
    close(committed,t.controls.cpu());
  }
  // An independent reverse program must read the actual saved invocation, even
  // after every public source buffer and parameter bank has been overwritten.
  DeviceProgram reverse(device);reverse.limit_workspace(64*1024*1024);
  auto saved=append_weighted_region_vjp(reverse,t,gradient,roots,error);reverse.finish();
  for(auto x:tensors(fields))x.fill_(NAN);
  for(auto bank:{weights.context,weights.query})for(auto x:tensors(bank))x.fill_(NAN);
  weights.context_owners.fill_(42);weights.query_owners.fill_(42);lengths.fill_(-42);times.fill_(-42);ids.fill_(-42);
  portable_torch::synchronize(device);reverse.run();
  require(error.cpu().item<int>()==0,"saved region reverse reread mutable source");compare(Fixture(1),c,t,saved);reverse.close();
  Fixture restored(1);auto src=tensors(restored.fields),dst=tensors(fields);
  for(int j=0;j<6;++j)dst[j].copy_(src[j]);
  ids.copy_(restored.ids);times.copy_(restored.times);lengths.copy_(restored.lengths);
  auto before=committed.cpu();ids[0][1].copy_(ids[0][0]);portable_torch::synchronize(device);p.run();
  require(error.cpu().item<int>()==2,"resident region accepted duplicate candidate IDs");close(committed,before);p.close();
  DeviceProgram refusal(device);bool rejected=false;
  try{append_weighted_region(refusal,fields,ids,times,lengths,weights,c,error,1);}catch(const std::invalid_argument&){rejected=true;}
  require(rejected,"resident region omitted scratch budget");refusal.close();
}
weighted::RegionalConfig config(const std::string& mode,bool attention=true,bool fixed=false,const std::string& regularizer="quad"){
  weighted::RegionalConfig c(mode);c.hidden_width=3;
  if(c.allocation_temperature)c.allocation_temperature=.8;if(c.regularizer){c.regularizer=regularizer;if(regularizer=="ent")c.ent_alpha=1.5;}
  if(attention){c.region_score="att";c.selection_width=2;c.region_key="content_stats";c.query_weight=c.regional()?"received":"mu";c.alpha_loc=.7;c.alpha_sel=.5;}
  if(c.regional()&&fixed){c.region_control_mode="fixed";c.region_evidence.reset();if(c.total())c.total_fraction=.35;else c.eta=.12;}
  c.validate();return c;
}
}
int main(int argc,char** argv){
  portable_torch::RuntimeSession runtime;
  try{
    auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("resident region requires explicit accelerator/FP32");
    auto device=portable_torch::resolve_device(args);at::set_num_threads(1);at::set_num_interop_threads(1);
    for(const auto& mode:{"regional_total","regional_total_zero","regional_threshold","regional_threshold_hard","constrained",
                         "hard_selection","direct_attention","unit_attention","pooled_attention","proportional"})check(device,config(mode));
    check(device,config("regional_total",false));check(device,config("regional_threshold",true,true));
    check(device,config("regional_total",true,true,"bern"));check(device,config("regional_total",true,false,"ent"));
    std::cout<<"weighted-region: passed configs=14 physical_rows=140 logical_frames=112 complete_fields_and_parameters_vjp=true history_int64=true\n";return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
