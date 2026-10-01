#include "content_fixture.h"
#include "event_reverse.h"
#include "full_vjp_fixture.h"
#include "portable_torch/runtime.hpp"
#include <ATen/Parallel.h>
#include <torch/csrc/autograd/autograd.h>
#include <cmath>
#include <iostream>
#include <limits>

namespace {
using namespace tide;using namespace tide::device_online;
constexpr Index batches=2,steps=3,base=(Index(1)<<55)+17;
struct Fixture {test::Fixture f;Index width,heads,kv_heads,window;bool adopt,active,clear;};
Fixture fixture(Index width,int variant,at::ScalarType dtype) {
  Fixture s;s.width=width;s.heads=width==4?4:1;s.kv_heads=width==4?2:1;
  s.window=variant%2?1:3;s.adopt=variant!=1;s.active=variant==0||variant==3;s.clear=variant==3;
  auto& f=s.f;auto& g=f.graph;g.nodes={{0}};auto& n=g.nodes[0];
  n.memory="attention";n.full="identity";n.query_heads=s.heads;n.kv_heads=s.kv_heads;n.window=s.window;n.clear=s.clear;
  g.regions={{1,s.adopt,false,"content","positive-v1"}};g.inputs={0};g.outputs={0};g.compile();
  auto eye=at::eye(width,at::kFloat);const Index kv=width/s.heads*s.kv_heads;
  NodeWeights w{at::zeros({width},at::kFloat),eye,at::zeros({width},at::kFloat),at::full({width},s.active?1.f:-1.f)};
  w.extra={{"attn_q",eye*.211f},{"attn_k",eye.narrow(1,0,kv).clone()*.413f},
    {"attn_v",eye.narrow(1,0,kv).clone()*.717f},{"attn_out",eye*.351f}};
  f.model.nodes={w};f.model.input_scale={at::ones({},at::kFloat)};f.model.output_scale={at::ones({},at::kFloat)};
  test::model_dtype(f.model,dtype);f.initial.identity=g.identity;f.initial.batch_size=batches;f.initial.cut=base;
  for(Index b=0;b<batches;++b) {
    const Index rows=b?0:std::min<Index>(2,s.window);
    f.initial.states[{b,0}]={at::zeros({width},at::TensorOptions().dtype(dtype)),base-1,rows,
      {{"key",(at::arange(rows*kv,at::kFloat).remainder(7).reshape({rows,s.kv_heads,width/s.heads})*.03113f).to(dtype)},
       {"value",(at::arange(rows*kv,at::kFloat).remainder(5).reshape({rows,s.kv_heads,width/s.heads})*.06113f).to(dtype)}}};
    for(Index t=0;t<steps;++t)f.input.push_back({b,0,t,base+t,
      (at::arange(width,at::kFloat).remainder(11)*.03113f+.151f+.017f*(t+b)).to(dtype)});
  }
  return s;
}
bool proposal_on(int mode){return mode==1||mode==4||mode==5;}
bool cache_on(int mode,bool key){return mode==4||mode==5||mode==(key?2:3);}
float root_value(int mode,bool key=false){return mode==4?0.f:key?.03125f:.0625f;}
Tensor quantize(const Tensor& x,bool half){return half?x+(x.detach().to(at::kHalf).to(x.scalar_type())-x.detach()):x;}
Tensor matmul(const Tensor& x,const Tensor& w,bool half) {
  auto y=at::matmul(x,w);if(!half)return y;
  return y+(at::matmul(x.detach().to(at::kHalf),w.detach().to(at::kHalf)).to(y.scalar_type())-y.detach());
}
std::vector<Tensor> reference(const Fixture& s,int mode,at::ScalarType dtype) {
  at::AutoGradMode enabled(true);const auto& f=s.f;const bool half=f.input[0].value.scalar_type()==at::kHalf;
  const Index w=s.width,h=s.heads,kh=s.kv_heads,d=w/h,kv=kh*d;
  std::vector<Tensor> x,parameters,keys,values,leaves,loss;
  auto leaf=[&](const Tensor& a){return a.to(dtype).clone().set_requires_grad(true);};
  for(const auto& item:f.input)x.push_back(leaf(item.value));
  for(const char* name:{"attn_q","attn_k","attn_v","attn_out"})parameters.push_back(leaf(f.model.nodes[0].extra.at(name)));
  for(Index b=0;b<batches;++b){keys.push_back(leaf(f.initial.states.at({b,0}).slots.at("key")));values.push_back(leaf(f.initial.states.at({b,0}).slots.at("value")));}
  for(const auto& group:{x,parameters,keys,values})leaves.insert(leaves.end(),group.begin(),group.end());
  auto mapping=at::arange(h,at::kLong).div(h/kh,"floor");
  for(Index b=0;b<batches;++b) {
    auto k=keys[b],v=values[b];
    for(Index time=0;time<steps;++time) {
      // Separate independent leaves preserve None for Q/O with key-only roots;
      // concatenating parameter leaves would invent connected-zero gradients.
      auto q=matmul(x[b*steps+time],parameters[0],half).reshape({h,1,d});
      auto nk=at::cat({k,matmul(x[b*steps+time],parameters[1],half).reshape({1,kh,d})});
      auto nv=at::cat({v,matmul(x[b*steps+time],parameters[2],half).reshape({1,kh,d})});
      const Index drop=std::max<Index>(0,nk.size(0)-s.window);nk=nk.narrow(0,drop,nk.size(0)-drop);nv=nv.narrow(0,drop,nv.size(0)-drop);
      auto key=nk.permute({1,2,0}).index_select(0,mapping),value=nv.permute({1,0,2}).index_select(0,mapping);
      auto prob=at::softmax(matmul(q,key,half)/std::sqrt(double(d)),-1);
      auto output=matmul(quantize(at::matmul(prob,value).reshape({w}),half),parameters[3],half);
      if(proposal_on(mode))loss.push_back(output.sum()*root_value(mode));
      if(s.adopt||s.active){k=nk;v=nv;}
      if(s.clear&&s.active){k=k.narrow(0,0,0).clone();v=v.narrow(0,0,0).clone();}
    }
    if(cache_on(mode,true))loss.push_back(k.sum()*root_value(mode,true));
    if(cache_on(mode,false))loss.push_back(v.sum()*root_value(mode));
  }
  if(loss.empty())return std::vector<Tensor>(leaves.size());
  return torch::autograd::grad({at::stack(loss).sum()},leaves,{},false,false,true);
}
void check(at::Device device,Index width,int variant,int mode,at::ScalarType dtype,bool prefill) {
  at::NoGradGuard guard;auto s=fixture(width,variant,dtype);auto& f=s.f;
  ContentLimits l;l.queue=32;l.arrivals=64;l.outputs=64;l.trace=64;l.kv_rows=s.window;
  l.kv_trace_rows=256;l.attention_chunk_rows=prefill?2:1;l.attention_key_rows=1;l.prefill=prefill;l.workspace_bytes=64*1024*1024;
  ContentFlow flow(f.graph,f.model,f.initial,device,l);flow.advance_device(f.input,base+steps);
  ReverseTape t;t.graph=&f.graph;t.state=flow.state_tape();t.full=flow.full_tape();
  auto tapes=flow.parameter_banks().attention;if(tapes.size()!=1)throw std::runtime_error("event test lost actual cache group");
  const auto& a=tapes[0];auto meta=t.state.metadata.cpu();const Index count=t.state.count.cpu().item<Index>(),capacity=meta.size(0);
  if(count!=batches*steps)throw std::runtime_error("event test lost public input events");
  auto floats=a.values.options(),bools=floats.dtype(at::kBool),longs=t.state.metadata.options();
  StateVjp state;state.proposal=at::full({capacity,width},std::numeric_limits<float>::quiet_NaN(),floats);
  state.proposal_connected=at::zeros({capacity},bools);state.content=at::empty({capacity,width},floats);state.content_connected=at::empty({capacity},bools);
  if(proposal_on(mode)){state.proposal.narrow(0,0,count).fill_(root_value(mode));state.proposal_connected.narrow(0,0,count).fill_(true);}
  CacheCotangents roots;auto lengths=a.lengths.cpu();
  for(bool key:{true,false})if(cache_on(mode,key)) {
    auto value=at::full(a.key.sizes(),std::numeric_limits<float>::quiet_NaN(),at::kFloat);
    for(Index b=0;b<batches;++b)value[b].narrow(0,0,lengths[b].item<Index>()).fill_(root_value(mode,key));
    (key?roots.key:roots.value)=value.to(device);(key?roots.key_connected:roots.value_connected)=at::ones({batches},bools);
  }
  auto parameters=at::empty({event_parameter_offsets(f.graph,width).back()},floats),on=at::empty({1,4},bools);
  auto error=at::zeros({1},floats.dtype(at::kInt)),range=at::tensor(std::vector<Index>{0,count},at::kLong).to(device);
  CannProgram p(device);p.limit_workspace(64*1024*1024);
  for(auto x:{state.content,state.content_connected,parameters,on})p.zero(x);
  auto reverse=prepare_event_reverse(p,t,a,roots,error,64*1024*1024);
  append_event_reverse(p,t,a,reverse,range,state,parameters,on,error,prefill?2:1,64*1024*1024);p.finish();
  for(int replay=0;replay<2;++replay) {
    portable_torch::synchronize(device);p.run();if(error.cpu().item<int>())throw std::runtime_error("event reverse refused valid actual tape");
    auto dx=state.content.cpu(),connected=state.content_connected.cpu(),dp=parameters.cpu(),pc=on.cpu();
    auto dk=reverse.cache.key.cpu(),dv=reverse.cache.value.cpu(),kc=reverse.cache.key_connected.cpu(),vc=reverse.cache.value_connected.cpu();
    if(dk.scalar_type()!=at::kFloat||dv.scalar_type()!=at::kFloat)throw std::runtime_error("cache adjoints must be FP32");
    for(auto ref_type:{at::kFloat,at::kDouble}) {
      const auto expected=reference(s,mode,ref_type);const bool half=dtype==at::kHalf;
      std::vector<bool> seen(batches*steps,false);
      for(Index row=0;row<count;++row) {
        const Index b=meta[row][0].item<Index>(),time=meta[row][2].item<Index>()-base,index=b*steps+time;
        if(b<0||b>=batches||time<0||time>=steps||seen.at(index))throw std::runtime_error("event identity changed");seen[index]=true;
        test::full_same_precision(dx[row],connected[row],expected[index],"event content",half);
      }
      Index offset=0;for(Index kind=0;kind<4;++kind) {
        const Index columns=kind==0||kind==3?width:width/s.heads*s.kv_heads;
        test::full_same_precision(dp.narrow(0,offset,width*columns).reshape({width,columns}),pc[0][kind],expected[batches*steps+kind],"event parameter",half);offset+=width*columns;
      }
      for(Index b=0;b<batches;++b) {
        const Index size=f.initial.states.at({b,0}).slots.at("key").size(0);
        test::full_same_precision(dk[b].narrow(0,0,size),kc[b],expected[batches*steps+4+b],"initial key",half);
        test::full_same_precision(dv[b].narrow(0,0,size),vc[b],expected[batches*steps+4+batches+b],"initial value",half);
        if(dk[b].narrow(0,size,a.capacity-size).count_nonzero().item<Index>()||dv[b].narrow(0,size,a.capacity-size).count_nonzero().item<Index>())throw std::runtime_error("initial cache padding acquired gradients");
      }
    }
  }
  p.close();flow.close();
}
void cache_bridge(at::Device device,Index width,at::ScalarType dtype) {
  at::NoGradGuard guard;const auto floats=at::TensorOptions().dtype(at::kFloat);
  EventAttentionTape a;a.samples=3;a.nodes={0};a.width=width;a.heads=a.kv_heads=1;a.capacity=3;
  a.key=at::full({3,3,1,width},std::numeric_limits<float>::quiet_NaN(),floats.dtype(dtype)).to(device);
  a.value=a.key.clone();a.lengths=at::tensor({2,0,1},at::kLong).to(device);
  CacheCotangents local;CacheGradient carry;carry.lengths=a.lengths.clone();
  const std::vector<std::vector<Index>> flags={{1,0,1},{0,1,1},{1,1,0},{1,0,1}};
  std::vector<Tensor> host;
  for(Index side=0;side<4;++side) {
    auto value=at::full({3,3,1,width},std::numeric_limits<float>::quiet_NaN(),floats);
    for(Index owner=0;owner<3;++owner)if(flags[side][owner]) {
      const Index length=owner==0?2:owner==2?1:0;
      // Beyond the half range, and sub-half increments: cache cotangents must
      // never be quantized merely because the forward cache is half.
      const float number=owner==2?0.f:side==0?65536.f:side==2?.125f:-.03125f;
      value[owner].narrow(0,0,length).fill_(number);
    }
    host.push_back(value);
    auto& root=side<2?local:static_cast<CacheCotangents&>(carry);
    (side%2?root.value:root.key)=value.to(device);
    (side%2?root.value_connected:root.key_connected)=at::tensor(flags[side],at::kLong).to(at::kBool).to(device);
  }
  for(int mode=0;mode<3;++mode) {
    auto error=at::zeros({1},floats.dtype(at::kInt)).to(device);CannProgram p(device);
    auto result=mode==2?append_cache_bridge(p,a,local,carry,error,1024*1024):
      append_cache_seed(p,a,mode==1?local:CacheCotangents{},error,1024*1024);
    p.finish();
    for(int replay=0;replay<2;++replay) {
      p.run();if(error.cpu().item<int>())throw std::runtime_error("cache bridge refused matching boundary");
      for(int which=0;which<2;++which) {
        auto value=(which?result.value:result.key).cpu(),on=(which?result.value_connected:result.key_connected).cpu();
        if(value.scalar_type()!=at::kFloat)throw std::runtime_error("cache bridge rounded cotangents");
        for(Index owner=0;owner<3;++owner) {
          auto expected=at::zeros({3,1,width},floats);const Index length=owner==0?2:owner==2?1:0;
          const bool left=mode>0&&flags[which][owner],right=mode==2&&flags[which+2][owner];
          if(left)expected.narrow(0,0,length).add_(host[which][owner].narrow(0,0,length));
          if(right)expected.narrow(0,0,length).add_(host[which+2][owner].narrow(0,0,length));
          if(on[owner].item<bool>()!=(left||right)||!at::equal(value[owner],expected))
            throw std::runtime_error("cache bridge changed None/zero, FP32 sum or poisoned padding");
        }
      }
    }
    p.close();
  }
  for(int invalid=0;invalid<3;++invalid) {
    auto wrong=a;auto extra=carry;
    if(invalid<2)wrong.lengths=at::tensor({invalid==0?-1:4,0,1},at::kLong).to(device);
    else extra.lengths=at::tensor({1,0,1},at::kLong).to(device);
    auto error=at::zeros({1},floats.dtype(at::kInt)).to(device);CannProgram p(device);
    append_cache_bridge(p,wrong,local,extra,error,1024*1024);p.finish();p.run();
    if(error.cpu().item<int>()!=2)throw std::runtime_error("cache bridge accepted incompatible lengths");p.close();
  }
  auto reject=[&](const CacheCotangents& root,int64_t budget) {
    auto error=at::zeros({1},floats.dtype(at::kInt)).to(device);CannProgram p(device);bool refused=false;
    try{append_cache_seed(p,a,root,error,budget);}catch(const std::invalid_argument&){refused=true;}
    if(!refused)throw std::runtime_error("cache bridge accepted invalid root or budget");
  };
  reject(local,1);auto wrong=local;wrong.key=wrong.key.to(at::kHalf);reject(wrong,1024*1024);
  wrong=local;wrong.key_connected=Tensor{};reject(wrong,1024*1024);
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||(args.dtype!=at::kFloat&&args.dtype!=at::kHalf))throw std::invalid_argument("event VJP requires explicit NPU FP32/FP16");
    args.allow_npu_float16=true;auto device=portable_torch::resolve_device(args);at::set_num_threads(1);at::set_num_interop_threads(1);int cases=0;
    for(int variant=0;variant<4;++variant)for(int mode=0;mode<6;++mode)for(bool prefill:{false,true}) {
      try{check(device,4,variant,mode,args.dtype,prefill);++cases;}
      catch(...){std::cerr<<"event VJP variant="<<variant<<" mode="<<mode<<" prefill="<<prefill<<'\n';throw;}
    }
    for(Index width:{1,7,257}){check(device,width,0,5,args.dtype,true);++cases;}
    for(Index width:{1,257})cache_bridge(device,width,args.dtype);
    std::cout<<"device-event-vjp: passed cases="<<cases<<" replays="<<cases*2
      <<" cache_bridge_cases=6 cache_bridge_replays=12 cache_refusals=12 CPU=FP32_FP64 actual_tapes=true GQA=true None_zero=true\n";
    runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
