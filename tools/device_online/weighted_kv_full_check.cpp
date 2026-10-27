#include "weighted_kv_full.h"
#include "portable_torch/runtime.hpp"
#include <torch/csrc/autograd/autograd.h>
#include <ATen/Parallel.h>
#include <iostream>
namespace {
using namespace tide;using namespace tide::device_online;
constexpr int64_t R=6,B=3,C=4,D=8,DS=3,F=5,clock=INT64_C(18014398509481987),last=INT64_MAX;
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void close(const Tensor& a,const Tensor& b){require(a.sizes()==b.sizes(),"KV Full result shape");if(!at::allclose(a.cpu(),b,2e-5,2e-6)){
  std::cerr<<a.cpu()<<" versus "<<b<<'\n';throw std::runtime_error("KV Full numeric mismatch");}}
Tensor values(at::IntArrayRef shape,float shift){int64_t n=1;for(auto x:shape)n*=x;return (at::arange(n,at::kFloat)*.13+shift).sin().reshape(shape)*.2;}
Tensor& field(State& s,const std::string& name){return name=="value"?s.value:s.slots.at(name);}
std::vector<Tensor> roles(const WeightedKVFullParameters& w){return {w.query,w.residual.out,w.residual.gate,w.residual.up,w.residual.down};}
std::vector<State> fixture(const WeightedRawLayout& l){
  std::vector<std::vector<int64_t>> stamps{{},{clock,clock+2},{0,2},{last-4,last-2},{0},{}};std::vector<State> states;
  for(int64_t r=0;r<R;++r){State s;s.last_time=stamps[r].empty()?-1:stamps[r].back();s.observations=stamps[r].size();s.slots["times"]=at::tensor(stamps[r],at::kLong);
    for(const auto& part:l.fields()){auto shape=part.shape;if(part.ragged)shape.insert(shape.begin(),stamps[r].size());auto x=values(shape,.1+r);
      if(part.name=="value")s.value=at::zeros_like(x);else s.slots[part.name]=part.name=="weights"?x.abs()+.1:x;}
    if(r==4)s.slots["weights"].zero_();states.push_back(s);}return states;
}
struct Formula {std::vector<Tensor> leaves,output;};
Formula reference(const weighted::KVConfig& c,const WeightedRawLayout& l,const std::vector<State>& old,const Tensor& times,
    const Tensor& input,const Tensor& owners,const Tensor& active,const WeightedKVFullParameters& w){
  at::AutoGradMode enabled(true);Formula f;auto states=old;std::vector<std::vector<Tensor>> params(5);
  for(int r=0;r<R;++r)for(const auto& part:l.fields()){auto x=field(states[r],part.name).to(at::kFloat).clone().set_requires_grad(true);field(states[r],part.name)=x;f.leaves.push_back(x);}
  for(int r=0;r<R;++r)f.leaves.push_back(input[r].clone().set_requires_grad(true));
  for(int role=0;role<5;++role)for(int bank=0;bank<B;++bank){auto x=roles(w)[role][bank].clone().set_requires_grad(true);params[role].push_back(x);f.leaves.push_back(x);}
  const auto dh=D/c.heads;
  for(int r=0;r<R;++r){if(!active[r].item<bool>()){f.output.push_back(at::zeros({D},at::kFloat));continue;}
    const auto bank=owners[r].item<int64_t>();const auto& s=states[r];auto u=f.leaves[5*R+r],stamps=s.slots.at("times");std::vector<int64_t> selected,ages;
    for(int i=0;i<stamps.numel();++i){const auto stamp=stamps[i].item<int64_t>(),time=times[r].item<int64_t>();
      if(stamp<=time&&(!c.window||time-stamp<c.window)){selected.push_back(i);ages.push_back(time-stamp);}}
    auto ids=at::tensor(selected,at::kLong),age=at::tensor(ages,at::kLong).to(at::kFloat),writes=s.slots.at("weights").index_select(0,ids),total=writes.sum();
    auto positive=at::nonzero(writes>0).flatten();std::vector<Tensor> reads;
    for(int64_t h=0;h<c.heads;++h){auto read=at::zeros({dh},at::kFloat);
      if(positive.numel()){auto q=at::mv(params[0][bank],u/(u.square().mean()+c.epsilon).sqrt()).slice(0,h*dh,(h+1)*dh);
        const auto frequencies=c.full_frequencies.empty()?std::vector<double>{}:c.full_frequencies[h];std::vector<Tensor> logits,features;
        for(int j=0;j<positive.numel();++j){auto i=positive[j].item<int64_t>(),source=ids[i].item<int64_t>();
          auto key=s.slots.at("keys")[source].slice(0,h*dh,(h+1)*dh),value=s.slots.at("values")[source].slice(0,h*dh,(h+1)*dh),matrix=at::eye(dh,at::kFloat);
          for(int64_t pair=0;pair<int64_t(frequencies.size());++pair){auto phase=-age[i]/float(c.position_scale)*float(frequencies[pair]);const auto a=dh-2*int64_t(frequencies.size())+2*pair;
            matrix[a][a]=phase.cos();matrix[a][a+1]=-phase.sin();matrix[a+1][a]=phase.sin();matrix[a+1][a+1]=phase.cos();}
          auto logit=(at::mv(matrix,key)*q).sum()/std::sqrt(dh);
          if(!c.distance_slopes.empty())logit=logit-age[i]/float(c.position_scale)*float(c.distance_slopes[h]);
          logits.push_back(logit+writes[i].log());features.push_back(value);}
        read=total.clamp_max(1)*(at::stack(logits).softmax(0).unsqueeze(1)*at::stack(features)).sum(0);
      }reads.push_back(read);
    }
    auto x=u+c.alpha_att*at::mv(params[1][bank],at::cat(reads));auto normal=x/(x.square().mean()+c.epsilon).sqrt();
    auto gate=at::mv(params[2][bank],normal),up=at::mv(params[3][bank],normal);
    f.output.push_back(x+c.alpha_ff*at::mv(params[4][bank],gate*gate.sigmoid()*up));
  }return f;
}
void compare(const WeightedRawLayout& l,const Formula& f,const Tensor& bars,const Tensor& roots,
    const WeightedKVFullTape& t,const WeightedKVFullGradient& g){
  at::AutoGradMode enabled(true);std::vector<Tensor> output,cotangents;
  for(int r=0;r<R;++r){close(t.output[r],f.output[r]);if(roots[r].item<bool>()&&f.output[r].requires_grad()){output.push_back(f.output[r]);cotangents.push_back(bars[r]);}}
  auto expected=torch::autograd::grad(output,f.leaves,cotangents,true,false,true);auto raw=g.raw_connected.cpu(),input=g.input_connected.cpu();
  for(int r=0;r<R;++r){for(int j=0;j<5;++j){const auto& e=expected[5*r+j];const auto& part=l.fields()[j];require(raw[r][j].item<bool>()==e.defined(),"KV Full raw None/zero differs");
      if(e.defined())close(g.raw[r].slice(0,part.offset,part.offset+e.numel()).reshape(e.sizes()),e);}
    const auto& e=expected[5*R+r];require(input[r].item<bool>()==e.defined(),"KV Full private input root differs");
    if(e.defined())close(g.input[r],e);else require(g.input[r].cpu().eq(0).all().item<bool>(),"absent Full input derivative nonzero");}
  auto parameters=roles(g.parameters);std::vector<Tensor> flags{g.query_connected.cpu(),g.out_connected.cpu(),g.gate_connected.cpu(),g.up_connected.cpu(),g.down_connected.cpu()};
  for(int role=0;role<5;++role)for(int bank=0;bank<B;++bank){const auto& e=expected[6*R+role*B+bank];require(flags[role][bank].item<bool>()==e.defined(),"KV Full parameter role None/zero differs");
    if(e.defined())close(parameters[role][bank],e);else require(parameters[role][bank].cpu().eq(0).all().item<bool>(),"absent Full parameter derivative nonzero");}
}
void check(at::Device device,at::ScalarType dtype,int64_t horizon,bool positions){
  at::NoGradGuard ng;weighted::KVConfig c;c.width=D;c.summary_width=DS;c.ffn_width=F;c.heads=2;c.epsilon=.13;c.position_scale=7;c.window=horizon;
  c.alpha_att=.3;c.alpha_ff=positions?.7:.4;if(positions){c.full_frequencies={{.13},{-.07,.11}};c.distance_slopes={.01,.03};}
  WeightedRawLayout l(c,C);auto states=fixture(l);for(auto& state:states)for(const auto& part:l.fields())field(state,part.name)=field(state,part.name).to(dtype);
  auto source=encode_weighted_raw_state(l,states),raw=copy_weighted_raw_state(source,device);auto opt=raw.floating.options().dtype(at::kFloat);
  auto input=at::zeros({R,D},opt),times=at::zeros({R},opt.dtype(at::kLong)),owners=at::zeros_like(times),active=at::zeros({R},opt.dtype(at::kBool));
  auto bars=at::zeros_like(input),roots=at::zeros_like(active),error=at::zeros({1},opt.dtype(at::kInt));
  WeightedKVFullParameters w{at::zeros({B,D,D},opt),{at::zeros({B,D,D},opt),at::zeros({B,F,D},opt),at::zeros({B,F,D},opt),at::zeros({B,D,F},opt)}};
  DeviceProgram p(device);p.limit_workspace(64*1024*1024);auto t=append_weighted_kv_full(p,c,raw,times,input,owners,active,w,error);auto g=append_weighted_kv_full_vjp(p,t,bars,roots,error);p.finish();
  for(int pass=0;pass<2;++pass){auto u=values({R,D},.2+pass*.1),time=at::tensor(std::vector<int64_t>{0,clock+5,pass?clock:30,last,0,-99},at::kLong);u[5].fill_(NAN);
    auto ids=at::tensor(std::vector<int64_t>{0,0,pass?2:1,0,1,-99},at::kLong),present=at::tensor({1,1,1,1,1,0},at::kInt).to(at::kBool);
    WeightedKVFullParameters weights{values({B,D,D},.3+pass*.1),{values({B,D,D},.5),values({B,F,D},.7),values({B,F,D},.9),values({B,D,F},1.1)}};
    if(!pass)for(auto x:roles(weights))x[2].fill_(NAN);auto ref=reference(c,l,states,time,u,ids,present,weights);
    raw.floating.copy_(source.floating);raw.integer.copy_(source.integer);raw.floating[5].fill_(NAN);raw.integer[5].fill_(-99);
    input.copy_(u);times.copy_(time);owners.copy_(ids);active.copy_(present);auto host=roles(weights),target=roles(w);for(int j=0;j<5;++j)target[j].copy_(host[j]);
    for(int test=0;test<2;++test){auto flags=at::tensor({1,1,test,1,test,1},at::kInt).to(at::kBool),dy=values({R,D},.8);dy[3].zero_();dy[5].fill_(NAN);if(!test){dy[2].fill_(NAN);dy[4].fill_(NAN);}
      bars.copy_(dy);roots.copy_(flags);portable_torch::synchronize(device);p.run();require(!error.cpu().item<int>(),"KV Full rejected valid inputs");compare(l,ref,dy,flags,t,g);
      if(pass&&test){DeviceProgram reverse(device);auto saved=append_weighted_kv_full_vjp(reverse,t,bars,roots,error);reverse.finish();
        raw.floating.fill_(NAN);raw.integer.fill_(-99);times.fill_(-99);input.fill_(NAN);owners.fill_(-99);active.zero_();for(auto x:roles(w))x.fill_(NAN);
        portable_torch::synchronize(device);reverse.run();require(!error.cpu().item<int>(),"KV Full saved reverse reread source");compare(l,ref,dy,flags,t,saved);reverse.close();}
    }
  }
  active.fill_(true);portable_torch::synchronize(device);p.run();require(error.cpu().item<int>()!=0,"KV Full invalid source accepted");p.close();
  DeviceProgram refusal(device);bool denied=false;try{append_weighted_kv_full(refusal,c,raw,times,input,owners,active,w,error,1);}catch(const std::invalid_argument&){denied=true;}
  require(denied,"KV Full scratch budget omitted");refusal.close();
}
}
int main(int argc,char** argv){portable_torch::RuntimeSession runtime;
  try{auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||(args.dtype!=at::kFloat&&args.dtype!=at::kHalf))throw std::invalid_argument("KV Full requires explicit accelerator FP32/FP16 raw storage");
    args.allow_npu_float16=true;
    auto device=portable_torch::resolve_device(args);at::set_num_threads(1);at::set_num_interop_threads(1);
    for(int64_t horizon:{0,5})for(bool positions:{false,true})check(device,args.dtype,horizon,positions);
    std::cout<<"weighted-kv-full: passed configurations=4 replay_rows=96 heads=2 all_parameter_input_raw_vjp=true saved_versions=true compute_dtype=float32 adjoint_dtype=float32\n";return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
