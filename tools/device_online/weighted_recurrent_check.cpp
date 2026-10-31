#include "weighted_recurrent_formula.h"
#include "portable_torch/runtime.hpp"
#include <torch/csrc/autograd/autograd.h>
#include <ATen/Parallel.h>
#include <iostream>
namespace {
using namespace tide;using namespace tide::device_online;using namespace tide::device_online::recurrent_check;
constexpr int64_t R=6,B=3,C=4,clock=INT64_C(18014398509481987),last=INT64_MAX;
void require(bool ok,const std::string& s){if(!ok)throw std::runtime_error(s);}
void close(const Tensor& actual,const Tensor& expected){require(actual.sizes()==expected.sizes(),"recurrent shape differs");
  if(!at::allclose(actual.cpu(),expected,3e-5,3e-6)){std::cerr<<actual.cpu()<<" versus "<<expected<<'\n';throw std::runtime_error("recurrent numeric mismatch");}}
Tensor values(at::IntArrayRef shape,float shift){int64_t n=1;for(auto x:shape)n*=x;return (at::arange(n,at::kFloat)*.13+shift).sin().reshape(shape)*.2;}
std::vector<State> fixture(const weighted::RecurrentConfig& c,const WeightedRawLayout& l){
  std::vector<std::vector<int64_t>> stamps{{},{clock-2,clock},{0,2},{last-5,last-3},{0},{}};std::vector<State> states;
  for(int64_t r=0;r<R;++r){State s;s.last_time=stamps[r].empty()?-1:stamps[r].back();s.observations=stamps[r].size();s.slots["time"]=at::scalar_tensor(std::max<int64_t>(0,s.last_time),at::kLong);
    if(!c.basic())s.slots["conv_times"]=at::tensor(stamps[r],at::kLong);
    for(const auto& part:l.fields()){auto shape=part.shape;if(part.ragged)shape.insert(shape.begin(),stamps[r].size());auto x=values(shape,.1+r);
      if(part.name=="value")s.value=at::zeros_like(x);else s.slots[part.name]=part.name=="W"?at::tensor({1.f,2.f},at::kFloat):x;
      if(!r&&part.name!="value")s.slots[part.name].zero_();}
    states.push_back(s);}return states;
}
void compare(const WeightedRawLayout& l,const Formula& f,const std::map<std::string,WeightedRowAdjoint>& roots,
    const WeightedRecurrentTape& t,const WeightedRecurrentGradient& g){
  at::AutoGradMode enabled(true);std::vector<Tensor> output,bars;
  for(int64_t r=0;r<R;++r)for(const auto& [name,expected]:f.fields[r]){close(t.fields.at(name)[r].slice(0,0,expected.numel()),expected.to(t.fields.at(name).scalar_type()));
    const auto& root=roots.at(name);if(root.connected[r].item<bool>()&&expected.requires_grad()){output.push_back(expected);bars.push_back(root.value[r].slice(0,0,expected.numel()));}}
  if(t.update)require(at::equal(t.update->result.integer.cpu(),f.metadata),"recurrent raw clocks/cache differ");
  auto derivatives=output.empty()?std::vector<Tensor>(f.leaves.size()):torch::autograd::grad(output,f.leaves,bars,true,false,true);
  auto raw=g.raw.cpu(),raw_flags=g.raw_connected.cpu();
  for(size_t j=0;j<f.leaves.size();++j){const auto& label=f.labels[j];Tensor actual;bool connected;
    if(label.name.rfind("raw.",0)==0){const auto& field=l.field(label.name.substr(4));int64_t index=0;while(l.fields()[index].name!=field.name)++index;
      actual=raw[label.row].slice(0,field.offset,field.offset+f.leaves[j].numel()).reshape(f.leaves[j].sizes());connected=raw_flags[label.row][index].item<bool>();
    }else{const auto& part=label.parameter?g.parameters.at(label.name):(label.name=="mass"?g.mass:g.content);actual=part.value[label.row].cpu();connected=part.connected[label.row].cpu().item<bool>();}
    if(connected!=derivatives[j].defined()){std::cerr<<label.name<<" row="<<label.row<<" actual="<<connected<<" expected="<<derivatives[j].defined()<<'\n';throw std::runtime_error("recurrent None/zero differs");}
    try{if(connected)close(actual,derivatives[j]);else require(actual.eq(0).all().item<bool>(),"absent recurrent derivative nonzero");}
    catch(...){std::cerr<<label.name<<" row="<<label.row<<'\n';throw;}
  }
}
void check(at::Device device,at::ScalarType dtype,const std::string& family,const std::string& read,WeightedRecurrentAction action){
  at::NoGradGuard ng;weighted::RecurrentConfig c;c.memory_family=family;c.memory_read=read;c.width=3;c.summary_width=2;c.key_width=2;c.value_width=2;c.ffn_width=4;c.epsilon=.13;
  if(!c.basic()){c.heads=2;c.groups=1;c.conv_size=3;c.conv_stride=2;c.matrix_rates={.07,0};c.output_epsilon=.09;if(family=="gated_delta")c.qk_epsilon=.03;}
  if(family=="ssd"&&read=="state"){c.heads=4;c.groups=2;c.matrix_rates={.07,0,.11,0};}
  if(read=="state"){c.alpha_mem=0;c.alpha_ff=0;}c.validate();WeightedRawLayout l(c,c.basic()?0:C);auto states=fixture(c,l);for(auto& state:states)for(const auto& field:l.fields()){
    auto& value=field.name=="value"?state.value:state.slots.at(field.name);value=value.to(dtype);}
  auto packed=encode_weighted_raw_state(l,states);
  for(int64_t r=0;r<R;++r)for(const auto& field:l.fields())if(field.ragged)
    packed.floating[r].slice(0,field.offset+states[r].slots.at("conv_times").numel()*field.width,field.offset+C*field.width).fill_(NAN);
  auto raw=copy_weighted_raw_state(packed,device);auto options=raw.floating.options().dtype(at::kFloat);auto times=at::zeros({R},raw.integer.options()),mass=at::zeros({R},options),u=at::zeros({R,c.width},options);
  auto owners=at::zeros({R},raw.integer.options()),present=at::zeros({R},options.dtype(at::kBool)),error=at::zeros({1},options.dtype(at::kInt));WeightedRecurrentParameters weights;
  for(const auto& [name,shape]:weighted_recurrent_parameter_shapes(B,c))weights[name]=at::zeros(shape,options);
  DeviceProgram p(device);p.limit_workspace(192*1024*1024);
  const auto initial_bytes=p.retained_tensor_bytes();
  auto t=append_weighted_recurrent(p,c,action,raw,times,mass,u,owners,present,weights,error);
  // The component budget is its increment, excluding the program's existing
  // backend control storage (CUDA already owns an error flag at construction).
  const auto forward_bytes=p.retained_tensor_bytes()-initial_bytes;
  std::map<std::string,WeightedRowAdjoint> roots;for(const auto& [name,value]:t.fields)roots[name]={at::zeros(value.sizes(),options),at::zeros_like(present)};
  auto g=append_weighted_recurrent_vjp(p,t,roots,error);p.finish();
  for(int pass=0;pass<2;++pass){auto input=values({R,c.width},.3+pass*.1);input[5].fill_(NAN);if(!pass)input[4].fill_(NAN);
    auto receives=at::tensor({0.f,.2f,1.f,1.5f,.5f,NAN},at::kFloat),ids=at::tensor(std::vector<int64_t>{0,1,pass?2:0,0,1,-1},at::kLong);
    auto active=at::tensor({1,1,1,1,pass,0},at::kInt).to(at::kBool);
    auto ts=at::tensor(std::vector<int64_t>{0,pass&&action!=WeightedRecurrentAction::Update?clock-1:clock+2,last,last-1,5,-99},at::kLong);
    WeightedRecurrentParameters parameters;int j=0;for(const auto& [name,value]:weights){auto data=values(value.sizes(),.2+j++*.03+pass*.1);if(name=="key.matrix")data*=8;
      if(!pass)data[2].fill_(NAN);parameters[name]=data;}
    auto ref=formula(c,action,l,states,ts,receives,input,ids,active,parameters);
    raw.floating.copy_(packed.floating);raw.integer.copy_(packed.integer);times.copy_(ts);mass.copy_(receives);u.copy_(input);owners.copy_(ids);present.copy_(active);
    for(const auto& [name,value]:parameters)weights[name].copy_(value);
    for(size_t test=0;test<=roots.size();++test){std::map<std::string,WeightedRowAdjoint> host;size_t k=0;
      for(const auto& [name,root]:roots){auto flags=at::zeros({R},at::kBool),bar=at::full(root.value.sizes(),NAN,at::kFloat);
        for(int64_t r=0;r<R;++r)if(test==roots.size()?(r+k)%3!=0:test==k){flags[r]=true;
          if(action==WeightedRecurrentAction::Update||active[r].item<bool>())bar[r].copy_(r==3?at::zeros_like(bar[r]):values(bar[r].sizes(),.8));}
        root.value.copy_(bar);root.connected.copy_(flags);host[name]={bar,flags};++k;}
      portable_torch::synchronize(device);p.run();require(!error.cpu().item<int>(),"recurrent valid input rejected, code="+std::to_string(error.cpu().item<int>()));compare(l,ref,host,t,g);
      if(pass&&test==roots.size()){
        DeviceProgram reverse(device);auto saved=append_weighted_recurrent_vjp(reverse,t,roots,error);reverse.finish();
        raw.floating.fill_(NAN);raw.integer.fill_(-99);times.fill_(-99);mass.fill_(NAN);u.fill_(NAN);owners.fill_(-99);present.zero_();for(auto& [name,value]:weights)value.fill_(NAN);
        portable_torch::synchronize(device);reverse.run();require(!error.cpu().item<int>(),"recurrent saved VJP reread source");compare(l,ref,host,t,saved);reverse.close();
      }
    }
  }
  present.fill_(true);portable_torch::synchronize(device);p.run();require(error.cpu().item<int>()!=0,"recurrent invalid source accepted");p.close();
  DeviceProgram refusal(device);bool denied=false;try{append_weighted_recurrent(refusal,c,action,raw,times,mass,u,owners,present,weights,error,1);}catch(const std::invalid_argument&){denied=true;}
  require(denied,"recurrent tensor capacity omitted");refusal.close();
  if(family=="ssd"&&read=="state"&&action==WeightedRecurrentAction::Full){
    DeviceProgram total_refusal(device);denied=false;
    try{append_weighted_recurrent(total_refusal,c,action,raw,times,mass,u,owners,present,weights,error,forward_bytes-1);}
    catch(const std::invalid_argument& e){denied=std::string(e.what()).find("complete recurrent tensor budget")!=std::string::npos;}
    require(denied,"whole-composition tensor capacity omitted");total_refusal.close();}
  DeviceProgram rate_refusal(device);auto bad=c;bad.lambda_short=1e-300;denied=false;
  try{append_weighted_recurrent_snapshot(rate_refusal,bad,raw,times,mass,owners,present,B,false,error,256*1024*1024);}
  catch(const std::invalid_argument& e){denied=std::string(e.what()).find("short passive rate")!=std::string::npos;}
  require(denied,"positive short rate underflow accepted");rate_refusal.close();
  if(!c.basic()){DeviceProgram matrix_refusal(device);bad=c;bad.matrix_rates[0]=1e-300;denied=false;
    try{append_weighted_recurrent_snapshot(matrix_refusal,bad,raw,times,mass,owners,present,B,false,error,256*1024*1024);}
    catch(const std::invalid_argument& e){denied=std::string(e.what()).find("underflows FP32")!=std::string::npos;}
    require(denied,"positive matrix rate underflow accepted");matrix_refusal.close();}
}
void half_store(at::Device device,const std::string& family){
  at::NoGradGuard ng;weighted::RecurrentConfig c;c.memory_family=family;c.width=3;c.summary_width=2;c.key_width=2;c.value_width=2;
  if(!c.basic()){c.heads=2;c.groups=1;c.conv_size=2;}c.validate();WeightedRawLayout l(c,c.basic()?0:2);
  State state;state.slots["time"]=at::scalar_tensor(0,at::kLong);
  if(!c.basic())state.slots["conv_times"]=at::empty({0},at::kLong);
  for(const auto& part:l.fields()){auto shape=part.shape;if(part.ragged)shape.insert(shape.begin(),0);
    auto value=at::zeros(shape,at::kHalf);if(part.name=="value")state.value=value;else state.slots[part.name]=value;}
  state.slots.at("H").flatten()[0]=0x1p-24;
  auto initial=encode_weighted_raw_state(l,{state}),raw=copy_weighted_raw_state(initial,device),live=copy_weighted_raw_state(initial,device);
  auto f=raw.floating.options().dtype(at::kFloat);auto time=at::zeros({1},raw.integer.options()),owners=at::zeros_like(time);
  auto present=at::ones({1},f.dtype(at::kBool)),mass=at::zeros({1},f),input=at::ones({1,c.width},f),error=at::zeros({1},f.dtype(at::kInt));
  WeightedRecurrentParameters w;for(const auto& [name,shape]:weighted_recurrent_parameter_shapes(1,c))w[name]=at::zeros(shape,f);
  DeviceProgram p(device);p.limit_workspace(192*1024*1024);
  auto snapshot=append_weighted_recurrent_snapshot(p,c,raw,time,mass,owners,present,1,false,error,256*1024*1024);
  auto t=append_weighted_recurrent(p,c,WeightedRecurrentAction::Update,raw,time,mass,input,owners,present,w,error);
  auto bar=at::zeros({1,2},f);bar[0][1]=1;
  std::map<std::string,WeightedRowAdjoint> roots{{"W",{bar,present}}};auto g=append_weighted_recurrent_vjp(p,t,roots,error);
  append_weighted_raw_adopt(p,live,t.update->result,error);p.finish();portable_torch::synchronize(device);p.run();
  require(!error.cpu().item<int>(),"half recurrent zero receive rejected");
  require(at::equal(live.floating.cpu().view(at::kShort),initial.floating.view(at::kShort))&&at::equal(live.integer.cpu(),initial.integer),"half recurrent no-receive retention changed bits");
  require(snapshot.state.floating[0][l.field("H").offset].cpu().item<float>()==0x1p-24,"half recurrent snapshot flushed subnormal");
  require(!g.mass.connected.cpu().item<bool>(),"zero reception acquired a mass root");
  mass.fill_(0x1p-25);portable_torch::synchronize(device);p.run();require(!error.cpu().item<int>(),"half recurrent weak receive rejected");
  auto stored=decode_weighted_raw_state(l,copy_weighted_raw_state(live,at::kCPU)).at(0);
  require(stored.last_time==0&&stored.observations==1&&stored.slots.at("W").eq(0).all().item<bool>(),"half recurrent underflow lost update/clock or wrong rounding");
  if(!c.basic())require(stored.slots.at("conv_times").numel()==1,"half recurrent underflow lost raw-time cache entry");
  require(g.mass.connected.cpu().item<bool>()&&g.mass.value.cpu().item<float>()==1,"half recurrent rounded zero lost identity-cast mass VJP");
  auto before=copy_weighted_raw_state(live,at::kCPU);mass.fill_(1);w.at("summary.matrix").fill_(1e5);
  portable_torch::synchronize(device);p.run();require(error.cpu().item<int>()==24,"half recurrent store overflow was not refused");
  require(at::equal(live.floating.cpu().view(at::kShort),before.floating.view(at::kShort))&&at::equal(live.integer.cpu(),before.integer),"failed half recurrent store modified owner");p.close();
  DeviceProgram refusal(device);bool denied=false;roots.at("W").value=at::zeros({1,2},f.dtype(at::kHalf));
  try{append_weighted_recurrent_vjp(refusal,t,roots,error);}catch(const std::invalid_argument&){denied=true;}
  require(denied,"half recurrent cotangent accepted");refusal.close();
}
}
int main(int argc,char** argv){portable_torch::RuntimeSession runtime;
  try{auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||(args.dtype!=at::kFloat&&args.dtype!=at::kHalf))throw std::invalid_argument("recurrent requires explicit accelerator FP32/FP16 raw storage");
    args.allow_npu_float16=true;
    auto device=portable_torch::resolve_device(args);at::set_num_threads(1);at::set_num_interop_threads(1);
    for(const std::string family:{"affine","delta","gated_delta","ssd"})for(const std::string read:{"sum","state"})
      for(auto action:{WeightedRecurrentAction::Update,WeightedRecurrentAction::Memory,WeightedRecurrentAction::Full}){
        try{check(device,args.dtype,family,read,action);}catch(...){std::cerr<<family<<' '<<read<<" action="<<int(action)<<'\n';throw;}}
    if(args.dtype==at::kHalf)for(const std::string family:{"affine","delta","gated_delta","ssd"})half_store(device,family);
    std::cout<<"weighted-recurrent: passed configurations=24 complete_upd_mem_full=true independent_raw_fields=true all_parameter_input_vjp=true saved_versions=true compute_dtype=float32 adjoint_dtype=float32\n";return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
