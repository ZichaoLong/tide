#include "content_fixture.h"
#include "fiber_reverse.h"
#include "full_vjp_fixture.h"
#include "portable_torch/runtime.hpp"
#include <ATen/Parallel.h>
#include <torch/csrc/autograd/autograd.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>

namespace {
using namespace tide;using namespace tide::device_online;
constexpr Index samples=2,base=(Index(1)<<55)+17;
const std::array<Index,3> times{0,2,5};
const std::array<std::string,5> kinds{"sum","mean","linear","active-softmax","all-softmax"};
const std::array<std::string,6> names{"fiber_qkv","fiber_qkv_bias","fiber_out","fiber_out_bias","fiber_decay","fiber_pool"};
struct Fixture {test::Fixture f;Index width,heads;int pool;bool adopt,active,clear;};
Fixture fixture(Index width,int pool,int variant,at::ScalarType dtype) {
  Fixture s;s.width=width;s.heads=width==4?2:1;s.pool=pool;
  s.adopt=variant!=1;s.active=variant==0||variant==3;s.clear=variant==3;
  auto& f=s.f;auto& g=f.graph;g.nodes={{0}};auto& n=g.nodes[0];
  n.memory="lh-fiber-attention-"+kinds[pool]+"-repeat-v1";n.full="identity";n.clear=s.clear;n.query_heads=n.kv_heads=s.heads;
  g.regions={{1,s.adopt,false,"content","positive-v1"}};g.inputs={0,0,0,0};g.outputs={0};g.compile();
  g.source_domain->input={3,1,0,2};g.compile();
  auto eye=at::eye(width,at::kFloat);
  NodeWeights w{at::zeros({width},at::kFloat),eye,at::zeros({width},at::kFloat),at::full({width},s.active?1.f:-1.f)};
  w.extra={{names[0],at::cat({eye*.211f,eye*.413f,eye*.717f},1)},
    {names[1],at::arange(3*width,at::kFloat).remainder(7)*.00713f},{names[2],eye*.351f},
    {names[3],at::full({width},.03113f)},{names[4],at::full({},.01713f)}};
  if(pool>=2)w.extra[names[5]]=at::tensor({.123f,-.251f,.317f,0.f});
  f.model.nodes={w};for(float value:{.713f,.313f,0.f,1.137f})f.model.input_scale.push_back(at::full({},value));
  f.model.output_scale={at::ones({},at::kFloat)};test::model_dtype(f.model,dtype);
  f.initial.identity=g.identity;f.initial.batch_size=samples;f.initial.cut=base;
  for(Index b=0;b<samples;++b) {
    const Index rows=b?0:2;
    f.initial.states[{b,0}]={at::zeros({width},at::TensorOptions().dtype(dtype)),base-1,rows,
      {{"key",(at::arange(rows*width,at::kFloat).remainder(7).reshape({rows,s.heads,width/s.heads})*.03113f).to(dtype)},
       {"value",(at::arange(rows*width,at::kFloat).remainder(5).reshape({rows,s.heads,width/s.heads})*.06113f).to(dtype)},
       {"log_bias",(at::arange(rows,at::kFloat)*.07113f).to(dtype)}}};
    const std::array<Index,3> ports{3,0,2};
    for(Index t=0;t<3;++t)for(Index j=0;j<3-t;++j)f.input.push_back({b,ports[j],t,base+times[t],
      (at::arange(width,at::kFloat).remainder(11)*.03113f+.151f+.017f*(t+b+j)).to(dtype)});
  }
  return s;
}
bool proposal_on(int mode){return mode==1||mode>=5;}
bool cache_on(int mode,int which){return mode==2+which||mode>=5;}
float root(int mode,int which,bool half){return mode==6?0.f:(which+1)*.03125f*(half?256:1);}
Tensor quantize(const Tensor& x,bool half){return half?x+(x.detach().to(at::kHalf).to(x.scalar_type())-x.detach()):x;}
Tensor matmul(const Tensor& x,const Tensor& w,bool half) {
  auto y=at::matmul(x,w);if(!half)return y;
  return y+(at::matmul(x.detach().to(at::kHalf),w.detach().to(at::kHalf)).to(y.scalar_type())-y.detach());
}
std::vector<Tensor> reference(const Fixture& s,int mode,at::ScalarType dtype) {
  at::AutoGradMode enabled(true);const auto& f=s.f;const bool half=f.input[0].value.scalar_type()==at::kHalf;
  const Index w=s.width,h=s.heads,d=w/h;
  std::vector<Tensor> x,scales,parameters,keys,values,biases,leaves,loss;
  auto leaf=[&](const Tensor& a){return a.to(dtype).clone().set_requires_grad(true);};
  for(const auto& input:f.input)x.push_back(leaf(input.value));
  for(const auto& scale:f.model.input_scale)scales.push_back(leaf(scale));
  for(const auto& name:names)parameters.push_back(leaf(name==names[5]&&s.pool<2?at::zeros({4},at::kFloat):f.model.nodes[0].extra.at(name)));
  for(Index b=0;b<samples;++b) {
    const auto& slots=f.initial.states.at({b,0}).slots;
    keys.push_back(leaf(slots.at("key")));values.push_back(leaf(slots.at("value")));biases.push_back(leaf(slots.at("log_bias")));
  }
  for(const auto& group:{x,scales,parameters,keys,values,biases})leaves.insert(leaves.end(),group.begin(),group.end());
  for(Index b=0;b<samples;++b) {
    auto k=keys[b],v=values[b],bias=biases[b];Index last=base-1;
    for(Index time:times) {
      std::vector<Index> indices,slots;std::vector<Tensor> rows;
      for(size_t i=0;i<f.input.size();++i)if(f.input[i].batch==b&&f.input[i].time==base+time)indices.push_back(i);
      std::stable_sort(indices.begin(),indices.end(),[&](Index a,Index z){return f.graph.source_domain->input[f.input[a].port]<f.graph.source_domain->input[f.input[z].port];});
      for(Index i:indices){slots.push_back(f.graph.source_domain->input[f.input[i].port]);rows.push_back(quantize(x[i]*scales[f.input[i].port],half));}
      const Index count=rows.size();auto projected=quantize(matmul(at::stack(rows),parameters[0],half)+parameters[1],half).split(w,-1);
      auto q=projected[0].reshape({count,h,d}).transpose(0,1);
      auto nk=at::cat({k,projected[1].reshape({count,h,d})}),nv=at::cat({v,projected[2].reshape({count,h,d})}),nb=bias;
      if(k.size(0))for(Index tick=last;tick<base+time;++tick)nb=quantize(nb-parameters[4],half);
      nb=at::cat({nb,at::zeros({count},nb.options())});
      if(half)q=quantize(q*float(1./std::sqrt(double(d))),true);
      auto score=matmul(q,nk.permute({1,2,0}),half)/(half?1.:std::sqrt(double(d)));
      auto ys=quantize(at::matmul(at::softmax(score+nb,-1),nv.permute({1,0,2})).transpose(0,1).reshape({count,w}),half);
      auto ids=at::tensor(slots,at::kLong);Tensor pooled;
      if(s.pool==0)pooled=ys.sum(0);
      else if(s.pool==1)pooled=ys.sum(0)/count;
      else {auto c=s.pool==2?parameters[5].index_select(0,ids):s.pool==3?at::softmax(parameters[5].index_select(0,ids),0):at::softmax(parameters[5],0).index_select(0,ids);
        pooled=(ys*c.unsqueeze(1)).sum(0);}
      auto output=quantize(matmul(quantize(pooled,half),parameters[2],half)+parameters[3],half);
      if(proposal_on(mode))loss.push_back(output.sum()*root(mode,1,half));
      if(s.adopt||s.active){k=nk;v=nv;bias=nb;last=base+time;}
      if(s.clear&&s.active){k=k.narrow(0,0,0).clone();v=v.narrow(0,0,0).clone();bias=bias.narrow(0,0,0).clone();}
    }
    if(cache_on(mode,0))loss.push_back(k.sum()*root(mode,0,half));
    if(cache_on(mode,1))loss.push_back(v.sum()*root(mode,1,half));
    if(cache_on(mode,2))loss.push_back(bias.sum()*root(mode,2,half));
  }
  return loss.empty()?std::vector<Tensor>(leaves.size()):torch::autograd::grad({at::stack(loss).sum()},leaves,{},false,false,true);
}
void check(at::Device device,Index width,int pool,int variant,int mode,at::ScalarType dtype,bool prefill) {
  at::NoGradGuard guard;auto s=fixture(width,pool,variant,dtype);auto& f=s.f;const bool half=dtype==at::kHalf;
  ContentLimits l;l.queue=64;l.arrivals=128;l.outputs=64;l.trace=128;l.kv_rows=16;l.kv_trace_rows=512;
  l.attention_chunk_rows=prefill?3:1;l.attention_key_rows=2;l.prefill=prefill;l.workspace_bytes=256*1024*1024;
  ContentFlow flow(f.graph,f.model,f.initial,device,l);flow.advance_device(f.input,base+6);auto t=flow.reverse_tape();
  if(t.fiber.size()!=1)throw std::runtime_error("fiber reverse lost actual cache group");const auto& g=t.fiber[0];const auto& a=g.cache;
  auto meta=t.fiber_meta.cpu();const Index count=t.state.count.cpu().item<Index>(),capacity=t.state.metadata.size(0),fibers=t.fiber_count.cpu().item<Index>();
  if(count!=6||fibers!=Index(f.input.size()))throw std::runtime_error("fiber reverse lost actual source/event identities");
  auto floats=t.fiber_values.options(),booleans=floats.dtype(at::kBool);const Index budget=256*1024*1024;
  StateVjp state;state.proposal=at::full({capacity,width},std::numeric_limits<float>::quiet_NaN(),floats);
  state.proposal_connected=at::zeros({capacity},booleans);
  if(proposal_on(mode)){state.proposal.narrow(0,0,count).fill_(root(mode,1,half));state.proposal_connected.narrow(0,0,count).fill_(true);}
  CacheCotangents roots;auto lengths=a.lengths.cpu();
  for(int which=0;which<3;++which)if(cache_on(mode,which)) {
    auto value=at::full(which==2?g.bias.sizes():a.key.sizes(),std::numeric_limits<float>::quiet_NaN(),at::kFloat);
    for(Index b=0;b<samples;++b)value[b].narrow(0,0,lengths[b].item<Index>()).fill_(root(mode,which,half));
    (which==0?roots.key:which==1?roots.value:roots.bias)=value.to(device);
    (which==0?roots.key_connected:which==1?roots.value_connected:roots.bias_connected)=at::ones({samples},booleans);
  }
  auto error=at::zeros({1},floats.dtype(at::kInt)),stage=at::tensor({Index(0),count},at::kLong).to(device);
  if(half){bool refused=false;try{CannProgram bad(device);append_graph_vjp(bad,t,{},error,2,budget);}catch(const std::invalid_argument&){refused=true;}
    if(!refused)throw std::runtime_error("graph VJP accepted missing mandatory cotangents");}
  CannProgram p(device);p.limit_workspace(budget);auto links=append_reverse_links(p,t,error,budget/4);
  auto reverse=prepare_fiber_reverse(p,t,links,g,roots,error,budget);
  auto messages=at::empty({links.messages.size(0),width},floats),on=at::empty({links.messages.size(0)},booleans);
  auto partials=at::empty(t.fiber_values.sizes(),floats),parameters=at::empty({fiber_parameter_offsets(f.graph,width).back()},floats),pc=at::empty({1,6},booleans);
  for(auto x:{messages,on,partials,parameters,pc})p.zero(x);
  append_fiber_reverse(p,t,links,g,reverse,stage,state,messages,on,partials,parameters,pc,error,prefill?2:1,budget);p.finish();
  for(int replay=0;replay<2;++replay) {
    portable_torch::synchronize(device);p.run();if(error.cpu().item<int>())throw std::runtime_error("fiber reverse refused actual valid tape");
    auto dx=messages.cpu(),xc=on.cpu(),ds=partials.cpu(),dp=parameters.cpu(),connected=pc.cpu();
    const auto& cache=reverse.cache;std::vector<Tensor> values{cache.key.cpu(),cache.value.cpu(),cache.bias.cpu()};
    std::vector<Tensor> flags{cache.key_connected.cpu(),cache.value_connected.cpu(),cache.bias_connected.cpu()};
    for(auto type:{at::kFloat,at::kDouble}) {
      auto expected=reference(s,mode,type);std::vector<bool> seen(f.input.size(),false);
      auto dscale=at::zeros({4},at::kFloat),scale_on=at::zeros({4},at::kBool);
      for(Index row=0;row<fibers;++row) {
        const auto b=meta[row][0].item<Index>(),time=meta[row][2].item<Index>(),port=meta[row][4].item<Index>();
        auto it=std::find_if(f.input.begin(),f.input.end(),[&](const auto& input){return input.batch==b&&input.time==time&&input.port==port;});
        if(meta[row][3].item<Index>()!=0||it==f.input.end()||it->position!=meta[row][5].item<Index>())throw std::runtime_error("fiber source identity changed");
        const Index index=it-f.input.begin();if(seen[index])throw std::runtime_error("fiber source duplicated");seen[index]=true;
        test::full_same_precision(dx[row],xc[row],expected[index],"fiber physical source",half);
        dscale[port].add_(ds[row].sum());if(xc[row].item<bool>())scale_on[port].fill_(true);
      }
      const Index start=f.input.size();for(Index port=0;port<4;++port)test::full_same_precision(dscale[port],scale_on[port],expected[start+port],"physical source scale",half);
      Index offset=0;for(Index kind=0;kind<6;++kind) {
        auto shape=kind==5&&pool<2?std::vector<Index>{4}:f.model.nodes[0].extra.at(names[kind]).sizes().vec();
        Index size=1;for(Index d:shape)size*=d;
        auto got=kind==5&&pool<2?at::zeros(shape,at::kFloat):dp.narrow(0,offset,size).reshape(shape);
        test::full_same_precision(got,connected[0][kind],expected[start+4+kind],names[kind].c_str(),half);offset+=kind==5&&pool<2?0:size;
      }
      for(Index which=0;which<3;++which)for(Index b=0;b<samples;++b) {
        const Index size=f.initial.states.at({b,0}).slots.at("key").size(0);
        if(values[which].scalar_type()!=at::kFloat)throw std::runtime_error("fiber cache adjoints were rounded");
        test::full_same_precision(values[which][b].narrow(0,0,size),flags[which][b],expected[start+10+which*samples+b],"fiber initial cache",half);
        if(values[which][b].narrow(0,size,a.capacity-size).count_nonzero().item<Index>())throw std::runtime_error("fiber cache padding acquired adjoints");
      }
    }
  }
  p.close();flow.close();
}
void bias_bridge(at::Device device,at::ScalarType dtype) {
  at::NoGradGuard guard;FiberAttentionTape g;auto& a=g.cache;constexpr Index k=257;
  a.samples=3;a.nodes={0};a.width=a.heads=a.kv_heads=1;a.capacity=k;
  a.key=at::full({3,k,1,1},std::numeric_limits<float>::quiet_NaN(),at::TensorOptions().dtype(dtype)).to(device);a.value=a.key.clone();
  a.lengths=at::tensor({Index(2),Index(0),k},at::kLong).to(device);g.bias=a.key.reshape({3,k});
  CacheCotangents local;CacheGradient carry;carry.lengths=a.lengths.clone();
  local.bias=at::full({3,k},std::numeric_limits<float>::quiet_NaN(),at::kFloat);carry.bias=local.bias.clone();
  local.bias[0].narrow(0,0,2).fill_(65536.f);carry.bias[0].narrow(0,0,2).fill_(.125f);carry.bias[2].zero_();
  local.bias=local.bias.to(device);carry.bias=carry.bias.to(device);
  local.bias_connected=at::tensor({1,1,0},at::kLong).to(at::kBool).to(device);
  carry.bias_connected=at::tensor({1,0,1},at::kLong).to(at::kBool).to(device);
  for(int mode=0;mode<3;++mode) {
    auto error=at::zeros({1},a.lengths.options().dtype(at::kInt));CannProgram p(device);
    auto out=append_fiber_cache_seed(p,g,mode?local:CacheCotangents{},mode==2?&carry:nullptr,error,1024*1024);p.finish();
    for(int replay=0;replay<2;++replay) {
      p.run();if(error.cpu().item<int>())throw std::runtime_error("fiber cache boundary refused matching lengths");
      auto expected=at::zeros({3,k},at::kFloat);if(mode)expected[0].narrow(0,0,2).fill_(mode==2?65536.125f:65536.f);
      auto on=at::tensor(std::vector<Index>{mode!=0,mode!=0,mode==2},at::kLong).to(at::kBool);
      if(out.bias.scalar_type()!=at::kFloat||!at::equal(out.bias.cpu(),expected)||!at::equal(out.bias_connected.cpu(),on)
          ||out.key.cpu().count_nonzero().item<Index>()||out.value.cpu().count_nonzero().item<Index>())
        throw std::runtime_error("fiber bias boundary changed FP32 sum,None/zero or padding");
    }
    p.close();
  }
  for(int invalid=0;invalid<3;++invalid) {
    auto wrong=g;auto extra=carry;
    if(invalid<2)wrong.cache.lengths=at::tensor({invalid==0?Index(-1):k+1,Index(0),k},at::kLong).to(device);
    else extra.lengths=at::tensor({Index(1),Index(0),k},at::kLong).to(device);
    auto error=at::zeros({1},a.lengths.options().dtype(at::kInt));CannProgram p(device);
    append_fiber_cache_seed(p,wrong,local,&extra,error,1024*1024);p.finish();p.run();
    if(error.cpu().item<int>()!=2)throw std::runtime_error("fiber bias bridge accepted invalid lengths");p.close();
  }
  for(int invalid=0;invalid<3;++invalid) {
    auto wrong=local;if(invalid==0)wrong.bias=wrong.bias.to(at::kHalf);if(invalid==1)wrong.bias_connected=Tensor{};
    auto error=at::zeros({1},a.lengths.options().dtype(at::kInt));CannProgram p(device);bool refused=false;
    try{append_fiber_cache_seed(p,g,wrong,nullptr,error,invalid==2?1:1024*1024);}catch(const std::invalid_argument&){refused=true;}
    if(!refused)throw std::runtime_error("fiber bias bridge accepted invalid root/budget");
  }
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||(args.dtype!=at::kFloat&&args.dtype!=at::kHalf))throw std::invalid_argument("fiber reverse requires explicit NPU FP32/FP16");
    args.allow_npu_float16=true;auto device=portable_torch::resolve_device(args);at::set_num_threads(1);at::set_num_interop_threads(1);int cases=0;
    for(int pool=0;pool<5;++pool)for(int mode=0;mode<7;++mode)for(bool prefill:{false,true}) {
      try{check(device,4,pool,0,mode,args.dtype,prefill);++cases;}
      catch(...){std::cerr<<"fiber reverse pool="<<pool<<" mode="<<mode<<" prefill="<<prefill<<'\n';throw;}
    }
    for(int variant=1;variant<4;++variant)for(int mode:{0,5,6})for(bool prefill:{false,true}){check(device,4,4,variant,mode,args.dtype,prefill);++cases;}
    for(Index width:{1,257}){check(device,width,4,0,5,args.dtype,true);++cases;}
    bias_bridge(device,args.dtype);
    std::cout<<"device-fiber-reverse: passed cases="<<cases<<" replays="<<cases*2
      <<" bias_bridge_cases=3 bias_bridge_replays=6 cache_refusals=6 CPU=FP32_FP64 actual_tapes=true None_zero=true\n";
    runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
