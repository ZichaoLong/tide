// Raw tuple codec, independent timestamp-list formulas and all field roots.
#include "weighted_raw_update.h"
#include "portable_torch/runtime.hpp"
#include <torch/csrc/autograd/autograd.h>
#include <ATen/Parallel.h>
#include <iostream>
#include <limits>
namespace {
using namespace tide;using namespace tide::device_online;
constexpr int64_t clock=INT64_C(18014398509481987),last=std::numeric_limits<int64_t>::max();
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void close(const Tensor& a,const Tensor& b){
  require(a.sizes()==b.sizes()&&a.scalar_type()==b.scalar_type(),"raw comparison shape/type");
  if(!at::allclose(a.cpu(),b,1e-6,1e-7)){std::cerr<<a.cpu()<<" versus "<<b<<'\n';throw std::runtime_error("raw numeric mismatch");}
}
void exact(const Tensor& a,const Tensor& b){require(at::equal(a.cpu().contiguous().reshape({-1}).view(at::kByte),b.contiguous().reshape({-1}).view(at::kByte)),"raw exact bits changed");}
Tensor values(at::IntArrayRef shape,double shift,at::ScalarType dtype=at::kFloat){int64_t n=1;for(auto x:shape)n*=x;
  return (at::arange(n,at::kFloat)*.13+shift).sin().mul(.2).reshape(shape).to(dtype);}
WeightedRawLayout layout(const std::string& family){
  if(family=="kv"){weighted::KVConfig c;c.width=3;c.summary_width=2;c.window=5;return {c,4};}
  weighted::RecurrentConfig c;c.memory_family=family;c.width=3;c.summary_width=2;c.value_width=2;c.key_width=3;
  if(!c.basic()){c.heads=2;c.conv_size=3;c.conv_stride=2;}
  return {c,c.basic()?0:4};
}
Tensor& field(State& s,const std::string& name){return name=="value"?s.value:s.slots.at(name);}
const Tensor& field(const State& s,const std::string& name){return name=="value"?s.value:s.slots.at(name);}
std::vector<State> fixture(const WeightedRawLayout& l,at::ScalarType dtype=at::kFloat){
  std::vector<std::vector<int64_t>> stamps{{},{clock,clock+2},{0,2},{last-4,last-2}};
  std::vector<int64_t> clocks{-1,clock+2,2,last-2};std::vector<State> out;
  for(int row=0;row<4;++row){const int64_t n=l.capacity()?stamps[row].size():0;
    State s;s.last_time=clocks[row];s.observations=row?9:0;
    for(const auto& f:l.fields()){auto shape=f.shape;if(f.ragged)shape.insert(shape.begin(),n);
      auto x=values(shape,.3+row,dtype);if(f.name=="value"){x.zero_();x[0].fill_(-0.0);x[1].fill_(0x1p-24);s.value=x;}
      else {if(f.name=="W"||f.name=="weights")x=x.abs()+.1;s.slots[f.name]=x;}}
    if(!l.kv())s.slots["time"]=at::scalar_tensor(std::max<int64_t>(0,s.last_time),at::kLong);
    if(l.capacity())s.slots[l.kv()?"times":"conv_times"]=at::tensor(stamps[row],at::kLong);
    out.push_back(std::move(s));
  }return out;
}
void equal_states(const std::vector<State>& a,const std::vector<State>& b,bool bits){
  require(a.size()==b.size(),"raw state row count");
  for(size_t i=0;i<a.size();++i){require(a[i].last_time==b[i].last_time&&a[i].observations==b[i].observations&&a[i].slots.size()==b[i].slots.size(),"raw state clock/count/domain");
    auto compare=[&](const Tensor& x,const Tensor& y){if(bits||x.scalar_type()==at::kLong)exact(x,y);else close(x,y);};
    compare(a[i].value,b[i].value);for(const auto& [name,value]:a[i].slots)compare(value,b[i].slots.at(name));}
}
std::vector<State> converted(std::vector<State> states,at::ScalarType dtype){
  for(auto& s:states){s.value=s.value.to(dtype);for(auto& [name,x]:s.slots)if(x.is_floating_point())x=x.to(dtype);}return states;
}
void codec(const WeightedRawLayout& l){
  at::NoGradGuard ng;
  for(auto dtype:{at::kDouble,at::kFloat,at::kHalf}){auto states=fixture(l,dtype);auto packed=encode_weighted_raw_state(l,states);
    require(packed.integer[1][0].item<int64_t>()==clock+2&&packed.integer[3][0].item<int64_t>()==last-2,"raw time converted to float");
    equal_states(decode_weighted_raw_state(l,packed),states,true);
    // Padding is deliberately outside the decoding domain.
    for(int row=0;row<4;++row){const auto n=packed.integer[row][2].item<int64_t>();
      packed.integer[row].slice(0,4+n).fill_(-999);
      for(const auto& f:l.fields())if(f.ragged)packed.floating[row].slice(0,f.offset+n*f.width,f.offset+l.capacity()*f.width).fill_(NAN);}
    equal_states(decode_weighted_raw_state(l,packed),states,true);
    packed.integer[1][1].fill_(-1);bool refused=false;try{decode_weighted_raw_state(l,packed);}catch(const std::invalid_argument&){refused=true;}
    require(refused,"invalid raw checkpoint count accepted");
  }
  bool refused=false;try{encode_weighted_raw_state(l,fixture(l),1);}catch(const std::invalid_argument&){refused=true;}
  require(refused,"raw encoding byte budget omitted");
}
struct Reference {std::vector<State> states;std::vector<std::vector<Tensor>> old,proposal;std::vector<Tensor> mass;};
Reference reference(const WeightedRawLayout& l,const std::vector<State>& input,const Tensor& proposal,const Tensor& mass,const Tensor& time,const Tensor& present){
  at::AutoGradMode enable(true);Reference ref;
  for(size_t row=0;row<input.size();++row){State old=input[row];std::vector<Tensor> old_fields,new_fields;
    for(const auto& f:l.fields()){auto x=field(old,f.name).clone().set_requires_grad(true);field(old,f.name)=x;old_fields.push_back(x);
      new_fields.push_back(proposal[row].slice(0,f.offset,f.offset+f.width).clone().reshape(f.shape).set_requires_grad(true));}
    auto r=mass[row].clone().set_requires_grad(true);State next=old;
    if(present[row].item<bool>()&&r.item<float>()>0){const auto theta=time[row].item<int64_t>();auto w=r.clamp_max(1);
      next.last_time=theta;++next.observations;Tensor indices;
      if(l.capacity()){auto stamps=old.slots.at(l.kv()?"times":"conv_times");std::vector<int64_t> keep;
        for(int64_t i=0;i<stamps.numel();++i){const auto stamp=stamps[i].item<int64_t>();
          if(stamp<=theta&&(l.kv()?(!l.horizon()||theta-stamp<l.horizon()):theta-stamp<=l.horizon()))keep.push_back(i);}
        indices=at::tensor(keep,at::kLong);next.slots[l.kv()?"times":"conv_times"]=at::cat({stamps.index_select(0,indices),at::full({1},theta,at::kLong)});}
      if(!l.kv())next.slots["time"]=at::full({},theta,at::kLong);
      for(size_t j=1;j<l.fields().size();++j){const auto& f=l.fields()[j];auto value=new_fields[j];
        if(f.ragged){value=f.name=="weights"?w:(l.convolution()?w*value:value);
          next.slots[f.name]=at::cat({field(old,f.name).index_select(0,indices),value.unsqueeze(0)});}
        else next.slots[f.name]=value;
      }
    }
    ref.states.push_back(std::move(next));ref.old.push_back(std::move(old_fields));ref.proposal.push_back(std::move(new_fields));ref.mass.push_back(r);
  }return ref;
}
void gradients(const WeightedRawLayout& l,const Reference& ref,const Tensor& bars,const Tensor& flags,const WeightedRawUpdateGradient& g){
  at::AutoGradMode enable(true);auto a=g.old_values.cpu(),b=g.proposal.cpu(),m=g.mass.cpu(),af=g.old_connected.cpu(),bf=g.proposal_connected.cpu(),mf=g.mass_connected.cpu();
  const auto count=l.fields().size();
  for(int row=0;row<4;++row){std::vector<Tensor> outputs,cotangents,leaves=ref.old[row];leaves.insert(leaves.end(),ref.proposal[row].begin(),ref.proposal[row].end());leaves.push_back(ref.mass[row]);
    for(size_t j=0;j<count;++j)if(flags[row][j].item<bool>()){const auto& f=l.fields()[j];const auto& y=field(ref.states[row],f.name);
      outputs.push_back(y);cotangents.push_back(bars[row].slice(0,f.offset,f.offset+y.numel()).reshape(y.sizes()));}
    const auto expected=torch::autograd::grad(outputs,leaves,cotangents,true,false,true);
    for(int part=0;part<2;++part)for(size_t j=0;j<count;++j){const auto& value=expected[part*count+j];
      require((part?bf:af)[row][j].item<bool>()==value.defined(),"raw independent field None/zero differs");
      if(value.defined())close((part?b:a)[row].slice(0,l.fields()[j].offset,l.fields()[j].offset+value.numel()).reshape(value.sizes()),value);}
    require(mf[row].item<bool>()==expected.back().defined(),"raw receive root differs");if(expected.back().defined())close(m[row],expected.back());
  }
}
void device_check(const WeightedRawLayout& l,at::Device device,at::ScalarType dtype){
  at::NoGradGuard ng;auto states=fixture(l,dtype);auto raw=encode_weighted_raw_state(l,states),old=copy_weighted_raw_state(raw,device),owner=copy_weighted_raw_state(raw,device);
  const auto f=old.floating.options();auto proposal=at::zeros_like(old.floating),mass=at::zeros({4},f),times=at::zeros({4},f.dtype(at::kLong));
  auto present=at::zeros({4},f.dtype(at::kBool)),error=at::zeros({1},f.dtype(at::kInt));auto bars=at::zeros(proposal.sizes(),f.dtype(at::kFloat)),roots=at::zeros({4,int64_t(l.fields().size())},present.options());
  DeviceProgram p(device);p.limit_workspace(64*1024*1024);auto t=append_weighted_raw_update(p,l,old,times,mass,present,proposal,error);
  auto g=append_weighted_raw_update_vjp(p,t,bars,roots,error);append_weighted_raw_adopt(p,owner,t.result,error);p.finish();
  for(int pass=0;pass<2;++pass){
    if(pass&&l.capacity())for(const std::string name:{l.kv()?"times":"conv_times"}){
      states[1].slots[name]=states[1].slots[name].slice(0,1).clone();for(const auto& field:l.fields())if(field.ragged)states[1].slots[field.name]=states[1].slots[field.name].slice(0,1).clone();}
    raw=encode_weighted_raw_state(l,states);old.floating.copy_(raw.floating);old.integer.copy_(raw.integer);
    auto writes=at::full_like(raw.floating,NAN);for(int row=0;row<4;++row)for(size_t j=1;j<l.fields().size();++j){const auto& field=l.fields()[j];
      if(field.name!="weights")writes[row].slice(0,field.offset,field.offset+field.width).copy_(values({field.width},.2+row).abs());}
    auto r=at::tensor(std::vector<float>{pass?0.f:.3f,pass?.2f:1.f,0.f,1.5f},at::kFloat).to(dtype);
    auto theta=at::tensor(std::vector<int64_t>{0,clock+4,30,last},at::kLong),active=at::ones({4},at::kBool);
    if(pass){active[2]=false;r[2]=NAN;theta[2]=-1;}
    // Independent FP32 Jacobian at the actual quantized operands. The declared
    // cast VJP is identity in FP32; the forward oracle is rounded only on store.
    auto ref=reference(l,converted(states,at::kFloat),writes.to(at::kFloat),r.to(at::kFloat),theta,active);
    auto bar=at::full(raw.floating.sizes(),NAN,raw.floating.options().dtype(at::kFloat)),flags=at::zeros({4,int64_t(l.fields().size())},at::kBool);
    for(int row=0;row<4;++row)for(size_t j=0;j<l.fields().size();++j){flags[row][j]=(row+j+pass)%3!=0;if(!flags[row][j].item<bool>())continue;
      const auto& field=l.fields()[j];const auto size=::field(ref.states[row],field.name).numel();
      bar[row].slice(0,field.offset,field.offset+size).copy_(row==2?at::zeros({size},at::kFloat):values({size},.9));}
    proposal.copy_(writes);times.copy_(theta);mass.copy_(r);present.copy_(active);bars.copy_(bar);roots.copy_(flags);
    portable_torch::synchronize(device);p.run();require(!error.cpu().item<int>(),"raw device update rejected valid tuple");
    equal_states(decode_weighted_raw_state(l,copy_weighted_raw_state(owner,at::kCPU)),converted(ref.states,dtype),dtype==at::kHalf);gradients(l,ref,bar,flags,g);
    if(pass){
      DeviceProgram reverse(device);auto saved=append_weighted_raw_update_vjp(reverse,t,bars,roots,error);reverse.finish();
      old.integer.fill_(-99);old.floating.fill_(NAN);proposal.fill_(NAN);mass.fill_(NAN);times.fill_(-99);present.zero_();
      portable_torch::synchronize(device);reverse.run();require(!error.cpu().item<int>(),"raw saved reverse read overwritten sources");gradients(l,ref,bar,flags,saved);reverse.close();
    }
  }
  const auto before=copy_weighted_raw_state(owner,at::kCPU);portable_torch::synchronize(device);p.run();
  require(error.cpu().item<int>()==24,"raw malformed tuple not rejected");exact(owner.floating,before.floating);exact(owner.integer,before.integer);p.close();
  DeviceProgram refusal(device);bool denied=false;
  try{append_weighted_raw_update(refusal,l,old,times,mass,present,proposal,error,1);}catch(const std::invalid_argument&){denied=true;}
  require(denied,"raw update scratch capacity omitted");refusal.close();
}
void adoption_bits(at::Device device,at::ScalarType dtype){
  at::NoGradGuard ng;
  auto bits=dtype==at::kHalf
    ?at::tensor(std::vector<int16_t>{0,-32768,15360,32322,1},at::kShort).view(at::kHalf)
    :at::tensor(std::vector<int32_t>{0,INT32_MIN,1065353216,2143289410,1},at::kInt).view(at::kFloat);
  WeightedRawState source{bits.to(device),at::tensor(std::vector<int64_t>{clock,last,-1},at::kLong).to(device)};
  WeightedRawState target{at::full_like(source.floating,-3),at::full_like(source.integer,-42)};
  auto before=copy_weighted_raw_state(target,at::kCPU);auto error=at::ones({1},target.integer.options().dtype(at::kInt));
  DeviceProgram p(device);append_weighted_raw_adopt(p,target,source,error);p.finish();
  portable_torch::synchronize(device);p.run();exact(target.floating,before.floating);exact(target.integer,before.integer);
  error.zero_();portable_torch::synchronize(device);p.run();exact(target.floating,bits);exact(target.integer,source.integer.cpu());p.close();
}
void capacity(at::Device device,at::ScalarType dtype){
  at::NoGradGuard ng;weighted::KVConfig config;config.width=3;config.summary_width=2;WeightedRawLayout l(config,1);
  auto s=fixture(l,dtype);s.erase(s.begin()+1,s.end());s[0].last_time=0;s[0].observations=1;s[0].slots["times"]=at::zeros({1},at::kLong);
  for(const auto& f:l.fields())if(f.ragged)s[0].slots[f.name]=at::ones({1,f.width},at::TensorOptions().dtype(dtype)).reshape(f.shape.empty()?std::vector<int64_t>{1}:std::vector<int64_t>{1,f.width});
  const auto raw=encode_weighted_raw_state(l,s),old=copy_weighted_raw_state(raw,device),owner=copy_weighted_raw_state(raw,device);const auto opt=old.floating.options();
  auto time=at::ones({1},opt.dtype(at::kLong)),mass=at::ones({1},opt),yes=at::ones({1},opt.dtype(at::kBool)),error=at::zeros({1},opt.dtype(at::kInt));
  DeviceProgram p(device);auto t=append_weighted_raw_update(p,l,old,time,mass,yes,at::zeros_like(old.floating),error);
  append_weighted_raw_adopt(p,owner,t.result,error);p.finish();portable_torch::synchronize(device);p.run();
  require(error.cpu().item<int>()==25,"infinite logical history exceeded finite simultaneous capacity silently");exact(owner.floating,raw.floating);exact(owner.integer,raw.integer);
  error.zero_();mass.zero_();time.fill_(last);portable_torch::synchronize(device);p.run();
  require(!error.cpu().item<int>(),"zero receive at maximum clock wrongly exhausted cache");exact(owner.floating,raw.floating);exact(owner.integer,raw.integer);p.close();
}
}
int main(int argc,char** argv){portable_torch::RuntimeSession runtime;
  try{auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||(args.dtype!=at::kFloat&&args.dtype!=at::kHalf))throw std::invalid_argument("weighted raw check requires explicit device and FP32/FP16");
    const auto device=portable_torch::resolve_device(args);at::set_num_threads(1);at::set_num_interop_threads(1);
    for(const std::string family:{"kv","affine","delta","gated_delta","ssd"})codec(layout(family));
    std::cout<<"weighted-raw-codec: passed families=5 dtypes=3\n";
    if(device.is_cpu())return 0;
    for(const std::string family:{"kv","affine","delta","gated_delta","ssd"})device_check(layout(family),device,args.dtype);
    capacity(device,args.dtype);adoption_bits(device,args.dtype);
    std::cout<<"weighted-raw: passed families=5 codec_dtypes=3 replay_rows=40 independent_field_vjp=true saved_reverse=true exact_int64=true failed_no_commit=true adoption_bits=true payload_dtype="
      <<(args.dtype==at::kHalf?"float16":"float32")<<" adjoint_dtype=float32\n";return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
