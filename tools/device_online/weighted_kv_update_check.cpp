// Pinned KV §1.2–1.3; independent per-row equations/autograd, no candidate recurrence.
#include "weighted_kv_update.h"
#include "portable_torch/runtime.hpp"
#include <torch/csrc/autograd/autograd.h>
#include <ATen/Parallel.h>
#include <iostream>
namespace {
using namespace tide;using namespace tide::device_online;
constexpr int64_t R=6,B=3,C=4,clock=INT64_C(18014398509481987),last=INT64_MAX;
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void close(const Tensor& a,const Tensor& b){require(a.sizes()==b.sizes(),"KV update result/gradient shape");
  if(!at::allclose(a.cpu(),b,1e-5,1e-6)){std::cerr<<a.cpu()<<" versus "<<b<<'\n';throw std::runtime_error("KV update numeric difference");}}
void exact(const Tensor& a,const Tensor& b){require(at::equal(a.cpu().contiguous().reshape({-1}).view(at::kByte),b.contiguous().reshape({-1}).view(at::kByte)),"KV update exact state changed");}
Tensor values(at::IntArrayRef shape,float shift){int64_t n=1;for(auto x:shape)n*=x;return (at::arange(n,at::kFloat)*.13+shift).sin().reshape(shape)*.2;}
Tensor& field(State& s,const std::string& name){return name=="value"?s.value:s.slots.at(name);}
const Tensor& field(const State& s,const std::string& name){return name=="value"?s.value:s.slots.at(name);}
std::vector<State> fixture(const WeightedRawLayout& l,at::ScalarType dtype){
  const std::vector<std::vector<int64_t>> stamps{{},{clock,clock+2},{0,2},{last-4,last-2},{0},{}};std::vector<State> states;
  for(int64_t r=0;r<R;++r){State s;s.last_time=stamps[r].empty()?-1:stamps[r].back();s.observations=stamps[r].size();
    s.slots["times"]=at::tensor(stamps[r],at::kLong);
    for(const auto& f:l.fields()){auto shape=f.shape;if(f.ragged)shape.insert(shape.begin(),stamps[r].size());
      auto x=values(shape,.3+r).to(dtype);if(f.name=="value"){x.zero_();x[0].fill_(-0.0);s.value=x;}
      else s.slots[f.name]=f.name=="weights"?x.abs()+.1:x;}
    states.push_back(std::move(s));}return states;
}
std::vector<Tensor> tensors(const WeightedKVParameters& w){return {w.base,w.left,w.right,w.summary};}
WeightedKVParameters parameters(int64_t d,int64_t ds,int64_t rank,float shift){
  return {values({B,2*d,d},shift),values({B,2*d,rank},shift+.2),values({B,rank,d},shift+.4),values({B,ds,d},shift+.6)};
}
struct Formula {std::vector<State> output;std::vector<Tensor> leaves;};
Formula reference(const weighted::KVConfig& c,const WeightedRawLayout& l,const std::vector<State>& old,
    const Tensor& h,const Tensor& time,const Tensor& ids,const Tensor& present,const WeightedKVParameters& w){
  at::AutoGradMode enabled(true);Formula f;
  std::vector<std::vector<Tensor>> params(4);
  // Leaves: old fields row-major; complete private h rows; parameters role-major.
  for(int64_t r=0;r<R;++r){auto q=old[r];for(const auto& part:l.fields()){
      auto x=field(q,part.name).to(at::kFloat).clone().set_requires_grad(true);f.leaves.push_back(x);field(q,part.name)=x;}f.output.push_back(q);}
  for(int64_t r=0;r<R;++r)f.leaves.push_back(h[r].clone().set_requires_grad(true));
  auto roles=tensors(w);for(int role=0;role<4;++role)for(int64_t bank=0;bank<B;++bank){
    auto x=roles[role][bank].clone().set_requires_grad(true);params[role].push_back(x);f.leaves.push_back(x);}
  for(int64_t r=0;r<R;++r){auto& q=f.output[r];const auto input=f.leaves[R*5+r];
    if(!present[r].item<bool>()||input[0].item<float>()<=0)continue;
    const auto theta=time[r].item<int64_t>(),bank=ids[r].item<int64_t>();auto u=input.slice(0,1),z=u/(u.square().mean()+c.epsilon).sqrt();
    auto kv=at::mv(params[0][bank]+at::mm(params[1][bank],params[2][bank]),z),p=at::mv(params[3][bank],z);
    std::vector<int64_t> keep;auto stamps=q.slots.at("times");for(int64_t i=0;i<stamps.numel();++i){const auto t=stamps[i].item<int64_t>();
      if(t<=theta&&(!c.window||theta-t<c.window))keep.push_back(i);}
    auto indices=at::tensor(keep,at::kLong);q.slots["times"]=at::cat({stamps.index_select(0,indices),at::full({1},theta,at::kLong)});
    std::vector<Tensor> appended{kv.slice(0,0,c.width),kv.slice(0,c.width),p,input[0].clamp_max(1)};
    for(int j=1;j<5;++j){const auto& name=l.fields()[j].name;q.slots[name]=at::cat({q.slots.at(name).index_select(0,indices),appended[j-1].unsqueeze(0)});}
    q.last_time=theta;++q.observations;
  }return f;
}
void compare(const WeightedRawLayout& l,const Formula& f,const Tensor& bars,const Tensor& roots,
    const WeightedKVUpdateTape& t,const WeightedKVUpdateGradient& g){
  auto result=decode_weighted_raw_state(l,copy_weighted_raw_state(t.raw.result,at::kCPU));
  at::AutoGradMode enabled(true);std::vector<Tensor> outputs,cotangents;
  for(int64_t r=0;r<R;++r){require(result[r].last_time==f.output[r].last_time&&result[r].observations==f.output[r].observations,"KV update clock/count mismatch");
    exact(result[r].slots.at("times"),f.output[r].slots.at("times"));
    for(int j=0;j<5;++j){const auto& part=l.fields()[j];const auto& y=field(f.output[r],part.name);close(field(result[r],part.name),y.to(t.raw.result.floating.scalar_type()));
      if(roots[r][j].item<bool>()){outputs.push_back(y);cotangents.push_back(bars[r].slice(0,part.offset,part.offset+y.numel()).reshape(y.sizes()));}}}
  auto expected=torch::autograd::grad(outputs,f.leaves,cotangents,true,false,true),actual=tensors(g.parameters);
  auto old_flags=g.old_connected.cpu(),input_flags=g.content_connected.cpu(),kv_flags=g.kv_connected.cpu(),summary_flags=g.summary_connected.cpu();
  for(int64_t r=0;r<R;++r){for(int j=0;j<5;++j){const auto& dy=expected[r*5+j];const auto& part=l.fields()[j];
      require(old_flags[r][j].item<bool>()==dy.defined(),"KV old field None/zero mismatch");
      if(dy.defined())close(g.old_values[r].slice(0,part.offset,part.offset+dy.numel()).reshape(dy.sizes()),dy);}
    const auto& dy=expected[R*5+r];require(input_flags[r].item<bool>()==dy.defined(),"KV content None/zero mismatch");
    if(dy.defined())close(g.content[r],dy);else require(g.content[r].cpu().eq(0).all().item<bool>(),"absent KV content derivative not zero");}
  for(int role=0;role<4;++role)for(int64_t bank=0;bank<B;++bank){const auto& dy=expected[R*6+role*B+bank];
    require((role==3?summary_flags:kv_flags)[bank].item<bool>()==dy.defined(),"KV parameter role None/zero mismatch");
    if(dy.defined())close(actual[role][bank],dy);else require(actual[role][bank].cpu().eq(0).all().item<bool>(),"absent KV parameter derivative not zero");}
}
void check(at::Device device,at::ScalarType dtype){
  at::NoGradGuard ng;weighted::KVConfig c;c.width=3;c.summary_width=2;c.adapter_rank=2;c.window=5;c.epsilon=.13;
  WeightedRawLayout l(c,C);auto states=fixture(l,dtype);auto initial=encode_weighted_raw_state(l,states);
  for(int64_t r=0;r<R;++r)for(const auto& f:l.fields())if(f.ragged)
    initial.floating[r].slice(0,f.offset+states[r].slots.at("times").numel()*f.width,f.offset+C*f.width).fill_(NAN);
  auto old=copy_weighted_raw_state(initial,device),live=copy_weighted_raw_state(initial,device);
  const auto opt=old.floating.options().dtype(at::kFloat);auto h=at::zeros({R,c.width+1},opt),time=at::zeros({R},opt.dtype(at::kLong)),ids=at::zeros_like(time);
  auto present=at::zeros({R},opt.dtype(at::kBool)),bar=at::zeros(old.floating.sizes(),opt),roots=at::zeros({R,5},present.options()),error=at::zeros({1},opt.dtype(at::kInt));
  auto host_weights=parameters(c.width,c.summary_width,c.adapter_rank,.3);auto w=host_weights;w.base=w.base.to(device);w.left=w.left.to(device);w.right=w.right.to(device);w.summary=w.summary.to(device);
  DeviceProgram program(device);program.limit_workspace(64*1024*1024);
  auto tape=append_weighted_kv_update(program,c,old,h,time,ids,present,w,error);
  auto g=append_weighted_kv_update_vjp(program,tape,bar,roots,error);append_weighted_raw_adopt(program,live,tape.raw.result,error);program.finish();
  for(int pass=0;pass<2;++pass){auto input=values({R,c.width+1},.1+pass*.2);input.select(1,0).copy_(at::tensor(std::vector<float>{.3f,pass?0.f:1.f,pass?.0001f:0.f,1.5f,NAN,pass?1.f:-1.f}));
    auto selected=at::tensor(std::vector<int64_t>{0,pass?-99:1,pass?2:-99,1,-99,pass?0:-99},at::kLong);
    auto theta=at::tensor(std::vector<int64_t>{0,clock+5,30,last,-99,2},at::kLong),available=at::tensor({1,1,1,1,0,1},at::kInt).to(at::kBool);
    for(int r=0;r<R;++r)if(selected[r].item<int64_t>()<0)input[r].slice(0,1).fill_(NAN);
    auto weights=parameters(c.width,c.summary_width,c.adapter_rank,.3+pass*.2);if(!pass){weights.base[2].fill_(NAN);weights.left[2].fill_(NAN);weights.right[2].fill_(NAN);weights.summary[2].fill_(NAN);}
    auto ref=reference(c,l,states,input,theta,selected,available,weights);
    old.floating.copy_(initial.floating);old.integer.copy_(initial.integer);h.copy_(input);time.copy_(theta);ids.copy_(selected);present.copy_(available);
    auto source=tensors(weights),target=tensors(w);for(int role=0;role<4;++role)target[role].copy_(source[role]);
    for(int test=0;test<6;++test){auto flags=at::zeros({R,5},at::kBool),bars=at::full(initial.floating.sizes(),NAN,initial.floating.options().dtype(at::kFloat));
      for(int r=0;r<R;++r)for(int j=0;j<5;++j)if(test==5?(r+j+pass)%3!=0:j==test){flags[r][j]=true;const auto& part=l.fields()[j];const auto n=field(ref.output[r],part.name).numel();
        bars[r].slice(0,part.offset,part.offset+n).copy_(r==3?at::zeros({n},at::kFloat):values({n},.8));}
      bar.copy_(bars);roots.copy_(flags);portable_torch::synchronize(device);program.run();
      require(!error.cpu().item<int>(),"KV update rejected valid inputs");compare(l,ref,bars,flags,tape,g);
      if(pass==1&&test==5){DeviceProgram reverse(device);auto saved=append_weighted_kv_update_vjp(reverse,tape,bar,roots,error);reverse.finish();
        old.integer.fill_(-99);old.floating.fill_(NAN);h.fill_(NAN);ids.fill_(-99);time.fill_(-99);for(auto x:tensors(w))x.fill_(NAN);
        portable_torch::synchronize(device);reverse.run();require(!error.cpu().item<int>(),"KV reverse read changed source");compare(l,ref,bars,flags,tape,saved);reverse.close();}
    }
  }
  auto before=copy_weighted_raw_state(live,at::kCPU);portable_torch::synchronize(device);program.run();
  require(error.cpu().item<int>()!=0,"KV update accepted poisoned actual input");exact(live.floating,before.floating);exact(live.integer,before.integer);program.close();
  DeviceProgram refusal(device);bool denied=false;try{append_weighted_kv_update(refusal,c,old,h,time,ids,present,w,error,1);}catch(const std::invalid_argument&){denied=true;}
  require(denied,"KV update tensor capacity omitted");refusal.close();
}
void half_store(at::Device device){
  at::NoGradGuard ng;weighted::KVConfig c;c.width=1;c.summary_width=1;c.adapter_rank=1;WeightedRawLayout l(c,2);
  State s;s.value=at::zeros({2},at::kHalf);s.slots["times"]=at::empty({0},at::kLong);
  for(const std::string name:{"keys","values","summaries"})s.slots[name]=at::empty({0,1},at::kHalf);
  s.slots["weights"]=at::empty({0},at::kHalf);
  auto initial=encode_weighted_raw_state(l,{s}),old=copy_weighted_raw_state(initial,device),live=copy_weighted_raw_state(initial,device);
  auto f=old.floating.options().dtype(at::kFloat);auto h=at::ones({1,2},f);h[0][0]=0x1p-25;
  auto time=at::zeros({1},old.integer.options()),ids=at::zeros_like(time),present=at::ones({1},f.dtype(at::kBool)),error=at::zeros({1},f.dtype(at::kInt));
  WeightedKVParameters w{at::ones({1,2,1},f),at::zeros({1,2,1},f),at::ones({1,1,1},f),at::ones({1,1,1},f)};
  auto bar=at::zeros(old.floating.sizes(),f),roots=at::zeros({1,5},present.options());bar[0][l.field("weights").offset]=1;roots[0][4]=true;
  DeviceProgram p(device);auto t=append_weighted_kv_update(p,c,old,h,time,ids,present,w,error);
  auto g=append_weighted_kv_update_vjp(p,t,bar,roots,error);append_weighted_raw_adopt(p,live,t.raw.result,error);p.finish();portable_torch::synchronize(device);p.run();
  require(!error.cpu().item<int>(),"half KV store rejected weak positive reception");
  auto stored=decode_weighted_raw_state(l,copy_weighted_raw_state(live,at::kCPU)).at(0);
  require(stored.observations==1&&stored.last_time==0&&stored.slots.at("times").numel()==1
    &&stored.slots.at("weights").item<float>()==0,"half write underflow changed raw entry/time existence");
  require(g.content_connected.cpu().item<bool>()&&g.content[0][0].cpu().item<float>()==1
    &&!g.kv_connected.cpu().item<bool>()&&!g.summary_connected.cpu().item<bool>(),"half write cast VJP lost independent mass root");
  auto before=copy_weighted_raw_state(live,at::kCPU);h[0][0]=1;w.summary.fill_(1e5);
  portable_torch::synchronize(device);p.run();require(error.cpu().item<int>()==24,"half KV numerical overflow was not refused before commit");
  exact(live.floating,before.floating);exact(live.integer,before.integer);p.close();
}
}
int main(int argc,char** argv){portable_torch::RuntimeSession runtime;
  try{auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||(args.dtype!=at::kFloat&&args.dtype!=at::kHalf))throw std::invalid_argument("weighted KV update needs explicit accelerator and FP32/FP16 raw storage");
    args.allow_npu_float16=true;const auto device=portable_torch::resolve_device(args);at::set_num_threads(1);at::set_num_interop_threads(1);check(device,args.dtype);
    if(args.dtype==at::kHalf)half_store(device);
    std::cout<<"weighted-kv-update: passed replay_rows=72 banks=3 roots=5 full_parameter_chain=true none_zero=true saved_versions=true exact_int64=true failed_no_commit=true compute_dtype=float32 adjoint_dtype=float32\n";return 0;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}}
