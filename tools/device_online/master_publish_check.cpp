#include "content_fixture.h"
#include "device_optimizer.h"
#include "tide/stream.h"
#include "portable_torch/runtime.hpp"
#include "../../cpp/bench/streaming.h"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <algorithm>
#include <iostream>
#include <limits>

namespace {
using namespace tide;using namespace tide::device_online;
void require(bool x,const char* why){if(!x)throw std::runtime_error(why);}
Model copy_model(Model model,at::ScalarType dtype) {
  std::map<const void*,Tensor> copies;
  auto copy=[&](Tensor& x){auto [it,inserted]=copies.emplace(x.unsafeGetTensorImpl(),Tensor{});
    if(inserted)it->second=x.detach().to(dtype).clone();x=it->second;};
  for(auto& w:model.nodes){for(auto x:{&w.weight,&w.bias,&w.decay,&w.read})copy(*x);for(auto& [_,x]:w.extra)copy(x);}
  for(auto& w:model.regions)for(auto& [_,x]:w.extra)copy(x);
  for(auto* group:{&model.input_scale,&model.agg_scale,&model.edge_scale,&model.output_scale})for(auto& x:*group)copy(x);
  return model;
}
test::Fixture fixture(int kind,Index width,at::ScalarType dtype) {
  auto f=test::fixture(1,0);auto eye=at::eye(width,at::kFloat);
  const std::vector<std::string> pools{"sum","linear","active-softmax","all-softmax"};
  const std::vector<std::string> aggregates{"mean","weighted_mean","active_softmax","all_softmax"};
  for(Index n=0;n<4;++n) {
    auto& node=f.graph.nodes[n];auto& w=f.model.nodes[n];node.identity=false;
    node.readout="norm-fp32-v1";node.full="tanh";node.memory="ema";
    w.weight=eye*.5f;w.bias=at::zeros({width},at::kFloat);w.decay=at::zeros_like(w.bias);w.read=at::ones_like(w.bias);
    if(kind==1) {
      node.memory="lh-add-repeat-v1";w.extra["add_retention"]=at::full({},.5f);
      if(n!=1){node.full="swiglu";w.extra["ffn_gate"]=at::cat({eye*.5f,eye*-.25f},1);
        w.extra["ffn_up"]=at::cat({eye*.25f,eye*.5f},1);w.extra["ffn_down"]=at::cat({eye*.5f,eye*.25f},0);}
    }
    if(kind==2&&n!=1) {
      node.full=n%2?"lh-relu-layer-v1":"lh-silu-rms-v1";
      w.extra["lh_norm_weight"]=at::ones({width},at::kFloat);
      w.extra["lh_norm_bias"]=at::arange(width,at::kFloat).remainder(3)*.03125f;
    }
    if(kind==3) {
      node.memory="attention";node.query_heads=3;node.kv_heads=n%2?3:1;node.window=n%2?0:3;
      w.extra["attn_q"]=eye*.25f;w.extra["attn_k"]=eye.narrow(1,0,width/3*node.kv_heads)*.5f;
      w.extra["attn_v"]=eye.narrow(1,0,width/3*node.kv_heads).clone();w.extra["attn_out"]=eye*.5f;
    }
    if(kind==4) {
      node.memory="lh-fiber-attention-"+pools[n]+"-repeat-v1";node.query_heads=node.kv_heads=n%2?3:1;
      w.extra["fiber_qkv"]=at::cat({eye*.25f,eye*.5f,eye},1);
      w.extra["fiber_qkv_bias"]=at::arange(3*width,at::kFloat).remainder(7)*.015625f;
      w.extra["fiber_out"]=eye*.5f;w.extra["fiber_out_bias"]=at::full({width},.03125f);
      w.extra["fiber_decay"]=at::full({},.03125f);
      if(n>0)w.extra["fiber_pool"]=at::arange(f.graph.source_counts[n],at::kFloat)*.125f-.25f;
    }
    if(kind==5) {
      node.aggregation=aggregates[n];
      if(n>0)for(Index slot=0;slot<f.graph.source_counts[n];++slot)
        w.extra[(n==1?"agg_mass_":"agg_logit_")+std::to_string(slot)]=at::full({},.25f*slot-.5f);
    }
  }
  f.graph.compile();f.initial.identity=f.graph.identity;
  for(auto& [owner,s]:f.initial.states) {
    s.value=s.value.repeat({width/3}).to(dtype);
    if(kind==3||kind==4){const auto& node=f.graph.nodes[owner.second];const auto heads=node.kv_heads;
      s.slots={{"key",at::empty({0,heads,width/node.query_heads},dtype)},
               {"value",at::empty({0,heads,width/node.query_heads},dtype)}};
      if(kind==4)s.slots["log_bias"]=at::empty({0},dtype);}
  }
  for(auto& x:f.input)x.value=x.value.repeat({width/3}).to(dtype);
  f.model=copy_model(f.model,dtype);
  f.model.nodes[3].read=f.model.nodes[1].bias; // HARD Read still needs alias publication.
  f.model.input_scale[1]=f.model.agg_scale[0];f.model.output_scale[2]=f.model.edge_scale[1];
  return f;
}
std::vector<Tensor> bank_snapshot(const ParameterBanks& b) {
  std::vector<Tensor> result;
  auto add=[&](const Tensor& x){if(x.defined())result.push_back(x.cpu().clone());};
  for(auto x:{b.weights,b.biases,b.decay,b.retention,b.read,b.sources,b.emission,
      b.extra.lh_weights,b.extra.lh_biases,b.extra.gate,b.extra.up,b.extra.down,b.aggregate.weights,
      b.fiber.qkv,b.fiber.qkv_bias,b.fiber.projection,b.fiber.projection_bias,b.fiber.decay,b.fiber.pool})add(x);
  for(const auto& a:b.attention){add(a.qkv);add(a.projection);}return result;
}
void trajectory(at::Device device,at::ScalarType dtype,int kind,Index width,bool prefill,DeviceOptimizerKind algorithm) {
  auto f=fixture(kind,width,dtype);auto registry=f.model.parameters(false);
  auto cpu_model=copy_model(f.model,dtype);const auto cpu_payload=cpu_model.parameters(false);
  auto master_model=copy_model(f.model,at::kFloat);auto masters=master_model.parameters(false);
  auto layout=parameter_layout(f.graph,registry,width,device,16*1024*1024);
  OptimizerGroup group;group.lr=.0001;group.weight_decay=.0125;group.momentum=.5;group.amsgrad=true;group.eps=1e-5;
  for(const auto& o:registry.owners())group.parameters.push_back(o.canonical);
  std::unique_ptr<NamedOptimizer> reference;
  if(algorithm==DeviceOptimizerKind::sgd)reference=std::make_unique<SGD>(masters,std::vector<OptimizerGroup>{group});
  else reference=std::make_unique<AdamW>(masters,std::vector<OptimizerGroup>{group});
  DeviceOptimizer optimizer(layout,algorithm,{group},32*1024*1024);
  ContentLimits l;l.queue=96;l.arrivals=128;l.outputs=128;l.trace=1024;l.kv_rows=128;l.kv_trace_rows=8192;
  l.workspace_bytes=512*1024*1024;l.prefill=prefill;l.attention_key_rows=prefill?7:128;l.attention_chunk_rows=3;l.full_chunk_rows=3;
  ContentFlow flow(f.graph,f.model,f.initial,device,l);const auto banks=flow.parameter_banks();
  std::map<std::pair<Index,Index>,Index> positions;
  for(const auto& x:f.input)++positions[{x.batch,x.port}];
  auto error=at::zeros({1},layout.values.options().dtype(at::kInt));CannProgram update(device);
  optimizer.append_step(update,layout,error);append_parameter_publish(update,banks,layout,optimizer.values(),error,1024*1024);update.finish();
  auto q=f.initial;
  for(int step=0;step<4;++step) {
    std::vector<External> input;
    for(auto x:f.input){x.time+=step*8;x.position+=step*positions.at({x.batch,x.port});input.push_back(x);}
    auto expected=Streaming(f.graph,cpu_model,{}).run(q,input,(step+1)*8,(step+1)*8);
    auto actual=flow.advance(input,(step+1)*8);
    tide_bench::compare(actual,expected,true,dtype,std::nullopt,dtype==at::kHalf?2e-2:1e-5,dtype==at::kHalf?2e-3:1e-6);
    q=expected.continuation;
    // Synthetic public owner-gradient packet tests publication independently
    // of graph VJP. No CPU-produced gradient is supplied to the candidate.
    auto gradient=at::full({layout.values.numel()},std::numeric_limits<float>::quiet_NaN(),at::kFloat);
    auto flags=at::zeros({int64_t(layout.owners.size())},at::kBool);
    for(size_t i=0;i<layout.owners.size();++i) {
      const auto& o=layout.owners[i];auto master=masters.value(o.canonical);master.mutable_grad().reset();
      if(layout.offsets[i]<0||step==2||(i%3==0&&step!=3))continue;
      auto value=(at::arange(o.value.numel(),at::kFloat).remainder(7)-3)*.015625f;
      if(step==1)value.zero_();else value+=step*.0078125f;
      gradient.narrow(0,layout.offsets[i],value.numel()).copy_(value);flags[i].fill_(true);
      master.mutable_grad()=value.reshape(o.value.sizes());
    }
    layout.values.copy_(gradient);layout.connected.copy_(flags);
    portable_torch::synchronize(device);update.run();require(!error.cpu().item<int>(),"master publication update refused");
    reference->step();const auto updated=optimizer.values().cpu();
    for(size_t i=0;i<layout.owners.size();++i) {
      const auto& o=layout.owners[i];const auto value=masters.value(o.canonical);
      if(layout.offsets[i]>=0)require(at::allclose(updated.narrow(0,layout.offsets[i],value.numel()).reshape(value.sizes()),value,1e-5,1e-6),"FP32 master differs from independent CPU update");
      cpu_payload.value(o.canonical).copy_(value.to(dtype));
    }
    auto quantized=[&](const std::string& name) {
      for(size_t i=0;i<layout.owners.size();++i) {
        const auto& o=layout.owners[i];
        if(std::find(o.aliases.begin(),o.aliases.end(),name)!=o.aliases.end()) {
          if(layout.offsets[i]<0)return o.value;
          return updated.narrow(0,layout.offsets[i],o.value.numel()).reshape(o.value.sizes()).to(dtype);
        }
      }
      throw std::logic_error("unknown publication alias");
    };
    require(at::equal(banks.read[3].cpu(),quantized("nodes.3.read")),"HARD Read alias was not published at payload dtype");
    if(kind==5)for(Index n=1;n<4;++n)for(Index slot=0;slot<f.graph.source_counts[n];++slot)
      require(at::equal(banks.aggregate.weights[n][slot].cpu(),quantized("nodes."+std::to_string(n)+".extra."+(n==1?"agg_mass_":"agg_logit_")+std::to_string(slot)).to(at::kFloat)),"normalized bank bypassed payload rounding");
  }
  std::vector<External> suffix;for(auto x:f.input){x.time+=32;x.position+=4*positions.at({x.batch,x.port});suffix.push_back(x);}
  tide_bench::compare(flow.advance(suffix,40),Streaming(f.graph,cpu_model,{}).run(q,suffix,40,40),true,dtype,
    std::nullopt,dtype==at::kHalf?2e-2:1e-5,dtype==at::kHalf?2e-3:1e-6);
  const auto before=bank_snapshot(banks);error.fill_(7);CannProgram refused(device);
  append_parameter_publish(refused,banks,layout,at::full_like(optimizer.values(),.75f),error,1024*1024);refused.finish();
  portable_torch::synchronize(device);refused.run();const auto after=bank_snapshot(banks);
  require(error.cpu().item<int>()==7&&before.size()==after.size(),"publication overwrote prior error");
  for(size_t i=0;i<before.size();++i)require(at::equal(before[i].view(at::kByte),after[i].view(at::kByte)),"refused publication changed a forward bank");
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    bool smoke=false;std::vector<char*> argsv{argv[0]};
    for(int i=1;i<argc;++i)if(std::string(argv[i])=="--profile-smoke")smoke=true;else argsv.push_back(argv[i]);
    auto args=portable_torch::parse_cli(argsv.size(),argsv.data(),true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||(args.dtype!=at::kFloat&&args.dtype!=at::kHalf))throw std::invalid_argument("master publication requires explicit NPU FP32/FP16");
    args.allow_npu_float16=true;auto device=portable_torch::resolve_device(args);
    if(device.type()!=c10::DeviceType::PrivateUse1)throw std::invalid_argument("master publication requires NPU");
    at::set_num_threads(1);at::set_num_interop_threads(1);at::NoGradGuard guard;Index cases=0;
    for(int kind=0;kind<6;++kind)for(Index width:{3,33})for(bool prefill:{false,true})
      for(auto algorithm:{DeviceOptimizerKind::sgd,DeviceOptimizerKind::adamw}) {
        if(smoke&&(kind<4||width!=3||!prefill||algorithm!=DeviceOptimizerKind::sgd))continue;
        try{trajectory(device,args.dtype,kind,width,prefill,algorithm);++cases;}
        catch(...){std::cerr<<"publication kind="<<kind<<" width="<<width<<" prefill="<<prefill<<" optimizer="<<int(algorithm)<<'\n';throw;}
      }
    std::cout<<"master-publication: passed trajectories="<<cases<<" windows="<<cases*5
      <<" updates="<<cases*4<<" master=FP32 source=public_gradient_packets scope="
      <<(smoke?"profile-smoke":"publication_not_complete_training")<<'\n';
    runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
