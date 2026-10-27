#include "weighted_kv_query.h"
#include "portable_torch/runtime.hpp"
#include <torch/csrc/autograd/autograd.h>
#include <ATen/Parallel.h>
#include <iostream>
namespace {
using namespace tide;using namespace tide::device_online;
constexpr int64_t R=6,B=3,C=4,clock=INT64_C(18014398509481987),last=INT64_MAX;
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void close(const Tensor& a,const Tensor& b){require(a.sizes()==b.sizes(),"KV qry result shape");if(!at::allclose(a.cpu(),b,2e-5,2e-6)){
  std::cerr<<a.cpu()<<" versus "<<b<<'\n';throw std::runtime_error("KV qry numeric mismatch");}}
Tensor values(at::IntArrayRef shape,float shift){int64_t n=1;for(auto x:shape)n*=x;return (at::arange(n,at::kFloat)*.13+shift).sin().reshape(shape)*.2;}
Tensor& field(State& s,const std::string& name){return name=="value"?s.value:s.slots.at(name);}
std::vector<Tensor> roles(const WeightedKVQueryParameters& w){return {w.query,w.key,w.value};}
std::vector<State> fixture(const WeightedRawLayout& l,at::ScalarType dtype){
  std::vector<std::vector<int64_t>> stamps{{},{clock,clock+2},{0,2},{last-4,last-2},{0},{}};std::vector<State> states;
  for(int64_t r=0;r<R;++r){State s;s.last_time=stamps[r].empty()?-1:stamps[r].back();s.observations=stamps[r].size();s.slots["times"]=at::tensor(stamps[r],at::kLong);
    for(const auto& part:l.fields()){auto shape=part.shape;if(part.ragged)shape.insert(shape.begin(),stamps[r].size());auto x=values(shape,.1+r).to(dtype);
      if(part.name=="value")s.value=at::zeros_like(x);else s.slots[part.name]=part.name=="weights"?x.abs()+.1:x;}
    if(r==4)s.slots["weights"].zero_();states.push_back(s);}return states;
}
struct Formula {std::vector<Tensor> leaves;std::vector<std::array<Tensor,4>> output;};
Formula reference(const weighted::KVConfig& c,const WeightedRawLayout& l,const std::vector<State>& old,const Tensor& times,
    const Tensor& input,const Tensor& owners,const Tensor& present,const WeightedKVQueryParameters& w){
  at::AutoGradMode enabled(true);Formula f;std::vector<State> states=old;std::vector<std::vector<Tensor>> params(3);
  for(int64_t r=0;r<R;++r)for(const auto& part:l.fields()){auto x=field(states[r],part.name).to(at::kFloat).clone().set_requires_grad(true);field(states[r],part.name)=x;f.leaves.push_back(x);}
  for(int64_t r=0;r<R;++r)f.leaves.push_back(input[r].clone().set_requires_grad(true));
  auto parameters=roles(w);for(int role=0;role<3;++role)for(int bank=0;bank<B;++bank){auto x=parameters[role][bank].clone().set_requires_grad(true);params[role].push_back(x);f.leaves.push_back(x);}
  for(int64_t r=0;r<R;++r){std::array<Tensor,4> output{at::zeros({2},at::kFloat),at::zeros({2},at::kFloat),at::zeros({},at::kFloat),at::zeros({},at::kFloat)};
    if(present[r].item<bool>()){const auto& state=states[r];auto stamps=state.slots.at("times");std::vector<int64_t> selected,ages;
      for(int64_t i=0;i<stamps.numel();++i){auto stamp=stamps[i].item<int64_t>(),time=times[r].item<int64_t>();
        if(stamp<=time&&(!c.window||time-stamp<c.window)){selected.push_back(i);ages.push_back(time-stamp);}}
      auto ids=at::tensor(selected,at::kLong),age=at::tensor(ages,at::kLong).to(at::kFloat);auto writes=state.slots.at("weights").index_select(0,ids);
      for(int channel=0;channel<2;++channel){auto weight=channel?writes:writes*at::exp(-float(c.beta_short)*age);auto total=weight.sum();output[channel+2]=total;
        auto pos=at::nonzero(weight>0).flatten();if(!pos.numel())continue;const auto owner=owners[r].item<int64_t>();auto u=f.leaves[R*5+r];
        auto q=at::mv(params[0][owner],u/(u.square().mean()+c.epsilon).sqrt());std::vector<Tensor> logits,features;
        for(int64_t j=0;j<pos.numel();++j){auto i=pos[j].item<int64_t>();auto source=ids[i].item<int64_t>();
          auto key=at::mv(params[1][owner],state.slots.at("keys")[source]);auto value=at::mv(params[2][owner],state.slots.at("values")[source]);
          auto phase=-age[i]/float(c.position_scale)*float(c.read_frequencies[0]);auto cosine=phase.cos(),sine=phase.sin();
          auto rotated=at::stack({key[0]*cosine-key[1]*sine,key[0]*sine+key[1]*cosine});
          logits.push_back((rotated*q).sum()/std::sqrt(2.)+weight[i].log());features.push_back(value);}
        output[channel]=total.clamp_max(1)*(at::stack(logits).softmax(0).unsqueeze(1)*at::stack(features)).sum(0);
      }
    }f.output.push_back(output);
  }return f;
}
void compare(const WeightedRawLayout& l,const Formula& f,const Tensor& bars,const Tensor& stats,const Tensor& roots,
    const WeightedKVQueryTape& t,const WeightedKVQueryGradient& g){
  at::AutoGradMode enabled(true);std::vector<Tensor> outputs,cotangents;
  for(int64_t r=0;r<R;++r)for(int j=0;j<4;++j){const auto y=j<2?t.memory[r][j]:t.total[r][j-2];const auto& expected=f.output[r][j];close(y,expected);
    if(roots[r][j].item<bool>()&&expected.requires_grad()){outputs.push_back(expected);cotangents.push_back(j<2?bars[r][j]:stats[r][j-2]);}}
  auto expected=torch::autograd::grad(outputs,f.leaves,cotangents,true,false,true);auto raw_flags=g.raw_connected.cpu(),input_flags=g.input_connected.cpu();
  for(int64_t r=0;r<R;++r){for(int j=0;j<5;++j){const auto& e=expected[5*r+j];const auto& part=l.fields()[j];require(raw_flags[r][j].item<bool>()==e.defined(),"KV qry raw field root differs");
      if(e.defined())close(g.raw[r].slice(0,part.offset,part.offset+e.numel()).reshape(e.sizes()),e);}
    const auto& e=expected[R*5+r];require(input_flags[r].item<bool>()==e.defined(),"KV qry private input root differs");if(e.defined())close(g.input[r],e);else require(g.input[r].cpu().eq(0).all().item<bool>(),"absent qry input gradient nonzero");}
  auto gradients=roles(g.parameters);std::vector<Tensor> flags{g.query_connected.cpu(),g.key_connected.cpu(),g.value_connected.cpu()};
  for(int role=0;role<3;++role)for(int bank=0;bank<B;++bank){const auto& e=expected[R*6+role*B+bank];require(flags[role][bank].item<bool>()==e.defined(),"KV qry parameter role None/zero differs");
    if(e.defined())close(gradients[role][bank],e);else require(gradients[role][bank].cpu().eq(0).all().item<bool>(),"absent qry parameter derivative nonzero");}
}
void check(at::Device device,int64_t horizon,at::ScalarType dtype){
  at::NoGradGuard ng;weighted::KVConfig c;c.width=3;c.summary_width=2;c.memory_read="qry";c.read_frequencies={.13};c.position_scale=7;c.epsilon=.13;c.beta_short=.2;c.window=horizon;
  WeightedRawLayout l(c,C);auto states=fixture(l,dtype);auto source=encode_weighted_raw_state(l,states),raw=copy_weighted_raw_state(source,device);auto opt=raw.floating.options().dtype(at::kFloat);
  auto input=at::zeros({R,3},opt),times=at::zeros({R},opt.dtype(at::kLong)),owners=at::zeros_like(times),present=at::zeros({R},opt.dtype(at::kBool));
  auto bars=at::zeros({R,2,2},opt),stats=at::zeros({R,2},opt),roots=at::zeros({R,4},present.options()),error=at::zeros({1},opt.dtype(at::kInt));
  WeightedKVQueryParameters w{at::zeros({B,2,3},opt),at::zeros({B,2,3},opt),at::zeros({B,2,3},opt)};
  DeviceProgram program(device);program.limit_workspace(64*1024*1024);auto tape=append_weighted_kv_query(program,c,raw,times,input,owners,present,w,error);
  auto g=append_weighted_kv_query_vjp(program,tape,bars,stats,roots,error);program.finish();
  for(int pass=0;pass<2;++pass){auto u=values({R,3},.2+pass*.1),time=at::tensor(std::vector<int64_t>{0,clock+5,pass?clock:30,last,0,-99},at::kLong);
    auto ids=at::tensor(std::vector<int64_t>{-99,0,pass?2:1,0,-99,-99},at::kLong),active=at::tensor({1,1,1,1,1,0},at::kInt).to(at::kBool);
    u[0].fill_(NAN);u[4].fill_(NAN);u[5].fill_(NAN);WeightedKVQueryParameters weights{values({B,2,3},.3+pass*.1),values({B,2,3},.5),values({B,2,3},.7)};
    if(!pass)for(auto x:roles(weights))x[2].fill_(NAN);
    auto ref=reference(c,l,states,time,u,ids,active,weights);raw.floating.copy_(source.floating);raw.integer.copy_(source.integer);raw.floating[5].fill_(NAN);raw.integer[5].fill_(-99);
    times.copy_(time);input.copy_(u);owners.copy_(ids);present.copy_(active);auto host=roles(weights),target=roles(w);for(int j=0;j<3;++j)target[j].copy_(host[j]);
    for(int test=0;test<5;++test){auto flags=at::zeros({R,4},at::kBool),dy=at::full({R,2,2},NAN,at::kFloat),dw=at::full({R,2},NAN,at::kFloat);
      for(int r=0;r<R;++r)for(int j=0;j<4;++j)if(test==4?(r+j)%3!=0:j==test){flags[r][j]=true;if(j<2)dy[r][j].copy_(r==3?at::zeros({2},at::kFloat):values({2},.8));else dw[r][j-2]=r==3?0.f:.3f;}
      bars.copy_(dy);stats.copy_(dw);roots.copy_(flags);portable_torch::synchronize(device);program.run();require(!error.cpu().item<int>(),"KV qry rejected valid inputs");compare(l,ref,dy,dw,flags,tape,g);
      if(pass==1&&test==4){DeviceProgram reverse(device);auto saved=append_weighted_kv_query_vjp(reverse,tape,bars,stats,roots,error);reverse.finish();
        raw.floating.fill_(NAN);raw.integer.fill_(-99);times.fill_(-99);input.fill_(NAN);owners.fill_(-99);present.zero_();for(auto x:roles(w))x.fill_(NAN);
        portable_torch::synchronize(device);reverse.run();require(!error.cpu().item<int>(),"KV qry saved reverse reread source");compare(l,ref,dy,dw,flags,tape,saved);reverse.close();}
    }
  }
  present.fill_(true);portable_torch::synchronize(device);program.run();require(error.cpu().item<int>()!=0,"KV qry invalid source accepted");program.close();
  DeviceProgram refusal(device);bool denied=false;try{append_weighted_kv_query(refusal,c,raw,times,input,owners,present,w,error,1);}catch(const std::invalid_argument&){denied=true;}
  require(denied,"KV qry scratch budget omitted");refusal.close();
}
}
int main(int argc,char** argv){portable_torch::RuntimeSession runtime;
  try{auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||(args.dtype!=at::kFloat&&args.dtype!=at::kHalf))throw std::invalid_argument("KV qry requires explicit accelerator and FP32/FP16 raw storage");
    args.allow_npu_float16=true;auto device=portable_torch::resolve_device(args);at::set_num_threads(1);at::set_num_interop_threads(1);for(int64_t horizon:{0,5})check(device,horizon,args.dtype);
    std::cout<<"weighted-kv-query: passed horizons=2 replay_rows=120 independent_roots=4 full_input_parameter_raw_vjp=true saved_versions=true read_dtype=float32 adjoint_dtype=float32\n";return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
