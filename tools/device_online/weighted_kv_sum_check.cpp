#include "weighted_kv_sum.h"
#include "portable_torch/runtime.hpp"
#include <torch/csrc/autograd/autograd.h>
#include <ATen/Parallel.h>
#include <iostream>
namespace {
using namespace tide;using namespace tide::device_online;
constexpr int64_t R=6,C=4,clock=INT64_C(18014398509481987),last=INT64_MAX;
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void close(const Tensor& a,const Tensor& b){require(a.sizes()==b.sizes(),"KV sum shape mismatch");if(!at::allclose(a.cpu(),b,1e-5,1e-6)){
  std::cerr<<a.cpu()<<" versus "<<b<<'\n';throw std::runtime_error("KV sum numeric mismatch");}}
Tensor values(at::IntArrayRef shape,float shift){int64_t n=1;for(auto x:shape)n*=x;return (at::arange(n,at::kFloat)*.13+shift).sin().reshape(shape)*.2;}
Tensor& field(State& s,const std::string& name){return name=="value"?s.value:s.slots.at(name);}
std::vector<State> fixture(const WeightedRawLayout& l){
  std::vector<std::vector<int64_t>> stamps{{},{clock,clock+2},{0,2},{last-4,last-2},{0},{}};std::vector<State> states;
  for(int64_t r=0;r<R;++r){State s;s.last_time=stamps[r].empty()?-1:stamps[r].back();s.observations=stamps[r].size();s.slots["times"]=at::tensor(stamps[r],at::kLong);
    for(const auto& part:l.fields()){auto shape=part.shape;if(part.ragged)shape.insert(shape.begin(),stamps[r].size());auto x=values(shape,.1+r);
      if(part.name=="value")s.value=at::zeros_like(x);else s.slots[part.name]=part.name=="weights"?x.abs()+.1:x;}
    if(r==4)s.slots["weights"].zero_();states.push_back(s);}return states;
}
struct Formula {std::vector<Tensor> leaves,memory,total;};
Formula reference(const WeightedRawLayout& l,const std::vector<State>& states,const Tensor& times,const Tensor& present){
  at::AutoGradMode enabled(true);Formula f;
  for(int64_t r=0;r<R;++r){State state=states[r];for(const auto& part:l.fields()){
    auto x=field(state,part.name).clone().set_requires_grad(true);f.leaves.push_back(x);field(state,part.name)=x;}
    if(!present[r].item<bool>()){f.memory.push_back(at::zeros({2,2},at::kFloat));f.total.push_back(at::zeros({2},at::kFloat));continue;}
    std::vector<int64_t> selected,ages;auto stamps=state.slots.at("times");for(int64_t i=0;i<stamps.numel();++i){auto stamp=stamps[i].item<int64_t>(),time=times[r].item<int64_t>();
      if(stamp<=time&&(!l.horizon()||time-stamp<l.horizon())){selected.push_back(i);ages.push_back(time-stamp);}}
    auto ids=at::tensor(selected,at::kLong),age=at::tensor(ages,at::kLong).to(at::kFloat),p=state.slots.at("summaries").index_select(0,ids);
    auto weights=state.slots.at("weights").index_select(0,ids);std::vector<Tensor> ms,ws;
    for(int channel=0;channel<2;++channel){auto w=channel?weights:weights*at::exp(-.2*age);auto total=w.sum();ws.push_back(total);ms.push_back((w.unsqueeze(1)*p).sum(0)/total.clamp_min(1));}
    f.memory.push_back(at::stack(ms));f.total.push_back(at::stack(ws));
  }return f;
}
void compare(const WeightedRawLayout& l,const Formula& ref,const Tensor& bars,const Tensor& stats,const Tensor& roots,
    const WeightedKVSumTape& t,const WeightedKVWindowGradient& g){
  at::AutoGradMode enabled(true);std::vector<Tensor> outputs,cotangents;
  for(int64_t r=0;r<R;++r){close(t.memory[r],ref.memory[r]);close(t.total[r],ref.total[r]);for(int j=0;j<4;++j){
    auto value=j<2?ref.memory[r][j]:ref.total[r][j-2];if(roots[r][j].item<bool>()&&value.requires_grad()){
      outputs.push_back(value);cotangents.push_back(j<2?bars[r][j]:stats[r][j-2]);}}}
  auto expected=torch::autograd::grad(outputs,ref.leaves,cotangents,true,false,true);auto flags=g.connected.cpu();
  for(int64_t r=0;r<R;++r)for(int j=0;j<5;++j){const auto& dy=expected[5*r+j];const auto& part=l.fields()[j];
    require(flags[r][j].item<bool>()==dy.defined(),"KV sum raw independent root mismatch");
    if(dy.defined())close(g.raw[r].slice(0,part.offset,part.offset+dy.numel()).reshape(dy.sizes()),dy);
    else require(g.raw[r].slice(0,part.offset,part.offset+part.width*(part.ragged?l.capacity():1)).cpu().eq(0).all().item<bool>(),"KV sum absent raw derivative nonzero");}
}
void check(at::Device device,int64_t horizon){
  at::NoGradGuard ng;weighted::KVConfig c;c.width=3;c.summary_width=2;c.window=horizon;c.beta_short=.2;
  WeightedRawLayout layout(c,C);auto states=fixture(layout);auto source=encode_weighted_raw_state(layout,states),raw=copy_weighted_raw_state(source,device);
  auto time=at::zeros({R},raw.integer.options()),present=at::zeros({R},raw.integer.options().dtype(at::kBool)),error=at::zeros({1},raw.integer.options().dtype(at::kInt));
  auto bars=at::zeros({R,2,2},raw.floating.options()),stats=at::zeros({R,2},raw.floating.options()),roots=at::zeros({R,4},present.options());
  DeviceProgram program(device);program.limit_workspace(64*1024*1024);auto tape=append_weighted_kv_sum(program,c,raw,time,present,error);
  auto g=append_weighted_kv_sum_vjp(program,tape,bars,stats,roots,error);program.finish();
  for(int pass=0;pass<2;++pass){auto times=at::tensor(std::vector<int64_t>{0,clock+(pass?1:5),pass?clock:30,last,0,-99},at::kLong);
    auto selected=at::tensor({1,1,1,1,1,0},at::kInt).to(at::kBool);auto ref=reference(layout,states,times,selected);
    raw.floating.copy_(source.floating);raw.integer.copy_(source.integer);raw.floating[5].fill_(NAN);raw.integer[5].fill_(-99);time.copy_(times);present.copy_(selected);
    for(int test=0;test<5;++test){auto flags=at::zeros({R,4},at::kBool),dy=at::full({R,2,2},NAN,at::kFloat),dw=at::full({R,2},NAN,at::kFloat);
      for(int r=0;r<R;++r)for(int j=0;j<4;++j)if(test==4?(r+j)%3!=0:j==test){flags[r][j]=true;if(j<2)dy[r][j].copy_(r==2?at::zeros({2},at::kFloat):values({2},.8));else dw[r][j-2]=r==2?0.f:.3f;}
      bars.copy_(dy);stats.copy_(dw);roots.copy_(flags);portable_torch::synchronize(device);program.run();require(!error.cpu().item<int>(),"KV sum rejected valid snapshot");compare(layout,ref,dy,dw,flags,tape,g);
      if(pass==1&&test==4){DeviceProgram reverse(device);auto saved=append_weighted_kv_sum_vjp(reverse,tape,bars,stats,roots,error);reverse.finish();
        raw.floating.fill_(NAN);raw.integer.fill_(-99);time.fill_(-99);present.zero_();portable_torch::synchronize(device);reverse.run();
        require(!error.cpu().item<int>(),"KV sum saved reverse reread source");compare(layout,ref,dy,dw,flags,tape,saved);reverse.close();}
    }
  }
  present.fill_(true);portable_torch::synchronize(device);program.run();require(error.cpu().item<int>()==31,"KV sum invalid snapshot accepted");program.close();
  DeviceProgram denied(device);bool refused=false;try{append_weighted_kv_sum(denied,c,raw,time,present,error,1);}catch(const std::invalid_argument&){refused=true;}
  require(refused,"KV sum scratch budget omitted");denied.close();
}
}
int main(int argc,char** argv){portable_torch::RuntimeSession runtime;
  try{auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("KV sum check requires explicit accelerator/FP32");
    auto device=portable_torch::resolve_device(args);at::set_num_threads(1);at::set_num_interop_threads(1);
    for(int64_t horizon:{0,5})check(device,horizon);
    std::cout<<"weighted-kv-sum: passed horizons=2 replay_rows=120 independent_mem_roots=4 raw_vjp=true saved_versions=true exact_int64=true\n";return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
