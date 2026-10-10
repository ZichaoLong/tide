#include "device_backend.h"
#include "parameter_vjp.h"
#include "graph_vjp_fixture.h"
#include "full_vjp_fixture.h"
#include "tide/stream.h"
#include "portable_torch/runtime.hpp"
#include "../../cpp/bench/streaming.h"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <torch/csrc/autograd/autograd.h>
#include <iostream>
#include <limits>

namespace {
using namespace tide;
using namespace tide::device_online;
void require(bool on,const char* message){if(!on)throw std::runtime_error(message);}
void actual(at::Device device,int shape,Index width,bool prefill) {
  at::NoGradGuard guard;auto f=test::graph_vjp_fixture(shape,0,width);
  // Real TensorImpl aliases cross nodes, differentiable roles and HARD Read.
  f.model.nodes[2].weight=f.model.nodes[0].weight;
  f.model.nodes[2].bias=f.model.nodes[0].decay;
  f.model.nodes[3].read=f.model.nodes[0].bias;
  // Overlapping storage through a distinct TensorImpl is a distinct parameter.
  f.model.nodes[3].decay=f.model.nodes[0].bias.detach();
  f.model.input_scale[1]=f.model.agg_scale[0];f.model.output_scale[2]=f.model.edge_scale[1];
  if(shape==3)f.model.input_scale[0]=f.model.nodes[0].extra.at("add_retention");
  auto registry=f.model.parameters(false);const auto owners=registry.owners();std::vector<Tensor> leaves;
  for(const auto& owner:owners){owner.value.set_requires_grad(true);leaves.push_back(owner.value);}
  require(registry.canonical_name("nodes.0.bias")!=registry.canonical_name("nodes.3.decay"),"test collapsed overlapping parameters");
  ContentLimits limits;limits.prefill=prefill;limits.trace=512;limits.full_chunk_rows=3;
  if(width>3)limits.workspace_bytes=512*1024*1024;
  ContentFlow flow(f.graph,f.model,f.initial,device,limits);flow.advance_device(f.input,11);auto t=flow.reverse_tape();auto opts=t.fiber_values.options();
  GraphCotangents roots{at::full_like(t.outputs.values,.0625f),at::zeros_like(t.outputs.valid),at::full_like(t.pending.values,.015625f),
    at::zeros_like(t.pending.valid),at::full({2,4,width},.03125f,opts),at::zeros({2,4},opts.dtype(at::kBool))};
  auto error=at::zeros({1},opts.dtype(at::kInt));DeviceProgram reverse(device);reverse.limit_workspace(64*1024*1024);
  auto graph=append_graph_vjp(reverse,t,roots,error,3,128*1024*1024);reverse.finish();
  for(int mode=0;mode<3;++mode) {
    if(mode){roots.outputs_connected.copy_(t.outputs.valid);roots.pending_connected.copy_(t.pending.valid);roots.final_connected.fill_(true);}
    if(mode==2){roots.outputs.zero_();roots.pending.zero_();roots.final.zero_();}
    portable_torch::synchronize(device);reverse.run();require(!error.cpu().item<int>(),"graph alias fixture reverse failed");
    // The local reduction contract permits arbitrary poison in absent partials.
    const auto poison=std::numeric_limits<float>::quiet_NaN();
    if(graph.weights.defined()){graph.weights.masked_fill_(graph.full_connected.logical_not().view({4,1,1}),poison);
      graph.biases.masked_fill_(graph.full_connected.logical_not().view({4,1}),poison);}
    graph.decay.masked_fill_(graph.decay_connected.logical_not().view({4,1}),poison);
    graph.retention.masked_fill_(graph.retention_connected.logical_not(),poison);graph.scales.masked_fill_(graph.scale_connected.logical_not(),poison);
    DeviceProgram p(device);auto out=append_parameter_vjp(p,f.graph,registry,graph,error,16*1024*1024);p.finish();
    portable_torch::synchronize(device);p.run();require(!error.cpu().item<int>(),"valid parameter owner reduction refused");
    std::vector<Tensor> expected(owners.size());
    {
      at::AutoGradMode grad(true);Streaming cpu(f.graph,f.model,{});auto result=cpu.run(f.initial,f.input,11,11);
      tide_bench::compare(flow.result(),result,true,at::kFloat);
      if(mode){std::vector<Tensor> loss;
        for(const auto& x:result.outputs)loss.push_back(x.value.sum()*(mode==2?0.:.0625));
        for(const auto& [_,s]:result.continuation.states)loss.push_back(s.value.sum()*(mode==2?0.:.03125));
        for(const auto& a:result.continuation.pending)loss.push_back(a.value.sum()*(mode==2?0.:.015625));
        expected=torch::autograd::grad({at::stack(loss).sum()},leaves,{},false,false,true);}
    }
    auto values=out.values.cpu(),flags=out.connected.cpu();
    require(out.owners.size()==owners.size(),"registry owner count changed");
    for(size_t i=0;i<owners.size();++i) {
      require(out.owners[i].canonical==owners[i].canonical&&out.owners[i].aliases==owners[i].aliases,"registry identity changed");
      auto value=out.offsets[i]>=0?values.narrow(0,out.offsets[i],owners[i].value.numel()).reshape(owners[i].value.sizes()):at::zeros_like(owners[i].value);
      test::full_same(value,flags[i],expected[i],owners[i].canonical.c_str());
    }
    portable_torch::synchronize(device);p.run();require(at::equal(out.values.cpu(),values)&&at::equal(out.connected.cpu(),flags),"alias reduction replay accumulated stale gradients");
    if(shape==0&&width==3&&!prefill&&mode==0) {
      bool bounded=false;try{DeviceProgram small(device);append_parameter_vjp(small,f.graph,registry,graph,error,1);}catch(const std::invalid_argument&){bounded=true;}
      require(bounded,"parameter reduction budget refusal missing");
      ParameterRegistry wrong;wrong.add("nodes.0.weight",at::zeros({1},at::kFloat));bool rejected=false;
      try{DeviceProgram bad(device);append_parameter_vjp(bad,f.graph,wrong,graph,error,16*1024*1024);}catch(const std::invalid_argument&){rejected=true;}
      require(rejected,"wrong parameter alias shape accepted");
      ParameterRegistry empty;bool empty_bounded=false;
      try{DeviceProgram small(device);append_parameter_vjp(small,f.graph,empty,graph,error,1);}catch(const std::invalid_argument&){empty_bounded=true;}
      require(empty_bounded,"empty parameter registry bypassed minimum budget");
      DeviceProgram empty_program(device);auto none=append_parameter_vjp(empty_program,f.graph,empty,graph,error,1024);
      empty_program.finish();portable_torch::synchronize(device);empty_program.run();
      require(none.owners.empty()&&none.offsets.empty()&&!none.connected.cpu().any().item<bool>()&&
              !none.values.cpu().any().item<bool>(),"empty parameter registry fabricated an owner");
      ParameterRegistry subset;subset.add("nodes.0.weight",f.model.nodes[0].weight);
      subset.add("nodes.2.weight",f.model.nodes[2].weight);
      DeviceProgram selected(device);auto single=append_parameter_vjp(selected,f.graph,subset,graph,error,16*1024*1024);
      selected.finish();portable_torch::synchronize(device);selected.run();
      require(single.owners.size()==1&&!single.connected.cpu()[0].item<bool>(),"trainable subset lost alias or None identity");
    }
  }
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("parameter VJP gate requires explicit NPU FP32");
    const auto device=portable_torch::resolve_device(args);if(device.type()!=tide::device_online::resident_device_type)throw std::invalid_argument("parameter VJP gate requires NPU");
    at::set_num_threads(1);at::set_num_interop_threads(1);int cases=0;
    for(int shape:{0,3})for(Index width:{1,3,257})for(bool prefill:{false,true}) {
      try{actual(device,shape,width,prefill);cases+=3;}
      catch(...){std::cerr<<"parameter VJP shape="<<shape<<" width="<<width<<" prefill="<<prefill<<'\n';throw;}
    }
    std::cout<<"device-parameter-vjp: passed aliased_graph_cases="<<cases<<" poison=true overlapping_distinct=true replay=true scope=owner_adjoints_not_optimizer\n";
    runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
