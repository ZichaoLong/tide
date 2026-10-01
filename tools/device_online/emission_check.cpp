#include "content_fixture.h"
#include "content_profile.h"
#include "packed_emission.h"
#include "portable_torch/runtime.hpp"
#include "tide/stream.h"
#include "tide/greedy.h"
#include "../../cpp/bench/streaming.h"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using namespace tide;
using namespace tide::device_online;
void require(bool ok,const char* text){if(!ok)throw std::runtime_error(text);}
void check_tensor(const Tensor& a,const Tensor& b){require(b.scalar_type()==at::kLong?at::equal(a.cpu(),b):a.scalar_type()==b.scalar_type()&&at::allclose(a.cpu(),b,b.scalar_type()==at::kHalf?3e-3:1e-5,b.scalar_type()==at::kHalf?2e-3:1e-6),"emission value mismatch");}
void configure(test::Fixture& f,Index width) {
  auto& g=f.graph;g.compile();
  for(Index n=0;n<Index(g.nodes.size());++n) {
    auto& node=g.nodes[n];auto& w=f.model.nodes[n];
    const Index slots=g.outgoing_ports.offsets[n+1]-g.outgoing_ports.offsets[n];
    for(Index slot=0;slot<slots;++slot)if(!node.identity&&node.emission=="slot_affine") {
      auto matrix=at::eye(width,at::kFloat)*(.25f+.0625f*slot);
      if(width>1)matrix[0][width-1].fill_(.125f); // Non-symmetric: catches transposed weights.
      auto bias=at::full({width},.03125f*(slot+1),at::kFloat);
      w.extra["emit_w_"+std::to_string(slot)]=matrix;w.extra["emit_b_"+std::to_string(slot)]=bias;
    }
  }
  g.compile();f.initial.identity=g.identity;
}
test::Fixture component_fixture(Index width) {
  test::Fixture f;auto& g=f.graph;g.nodes={{0},{1},{2}};g.regions={{1},{1},{1}};
  g.edges={{0,1,2},{0,1,3},{2,1,1}};g.outputs={0,0,1,2};
  for(auto& n:g.nodes){n.memory="identity";n.full="identity";}
  g.nodes[0].emission="slot_affine";g.nodes[0].emit_period=3;g.nodes[0].emit_phases={-1,1,-2,0};
  g.nodes[2].identity=true; // Boundary adapters require unconditional broadcast.
  for(int n=0;n<3;++n)f.model.nodes.push_back({at::zeros({width},at::kFloat),at::eye(width,at::kFloat),
    at::zeros({width},at::kFloat),at::zeros({width},at::kFloat)});
  for(int e=0;e<3;++e){f.model.edge_scale.push_back(at::full({},e==1?0.f:-.5f,at::kFloat));f.model.agg_scale.push_back(at::ones({},at::kFloat));}
  for(int p=0;p<4;++p)f.model.output_scale.push_back(at::full({},p==0?0.f:.25f,at::kFloat));
  g.compile();g.layout->edge_source={3,2,1};g.layout->output={1,0,0,0};
  configure(f,width);return f;
}
Index components(at::Device device,at::ScalarType dtype) {
  Index cases=0;const Index base=(Index(1)<<55)/3*3;
  for(Index width:{1,7,33,257})for(Index chunk:{1,4}) {
    auto f=component_fixture(width);test::model_dtype(f.model,dtype);ContentProfile profile(f.graph,f.model,device);
    // Fault injection is below the public finite-parameter boundary. Whole
    // graph construction must continue to reject any nonfinite input model.
    profile.model.nodes[0].extra.at("emit_w_2").fill_(std::numeric_limits<float>::quiet_NaN());
    profile.model.nodes[0].extra.at("emit_b_2").fill_(std::numeric_limits<float>::quiet_NaN());
    PackedEmission emission(profile,device,2,16,16,chunk,64*1024*1024);
    auto coordinates=at::tensor(std::vector<Index>{0,0,base,base,0,0,base+1,base+1,1,1,base,base,1,2,base,base,-9,-9,-9,-9},at::kLong).reshape({5,4});
    auto x=(at::arange(5*width,at::kFloat).reshape({5,width}).remainder(11)-5)*.03125f;x=x.to(dtype);x[4].fill_(std::numeric_limits<float>::quiet_NaN());
    auto valid=at::ones({5},at::kBool);valid[4].fill_(false);
    ActionBatch action{coordinates.to(device),x.to(device),valid.to(device)};
    auto error=at::zeros({1},action.coordinates.options().dtype(at::kInt));
    CannProgram program(device);auto output=emission.append_stage(program,action,error);program.finish();
    for(Index replay=0;replay<2;++replay) {
      if(replay){coordinates[0][2].fill_(base+2);coordinates[0][3].fill_(base+2);valid[2].fill_(false);}
      action.coordinates.copy_(coordinates);action.valid.copy_(valid);program.run();
      require(error.cpu().item<int>()==0,"valid packed emission refused");
      const auto count=output.count.cpu().item<Index>();auto meta=output.meta.cpu(),values=output.values.cpu();
      auto edges=output.arrivals.coordinates.cpu(),ev=output.arrivals.values.cpu(),on=output.arrivals.valid.cpu();
      auto ports=output.outputs.coordinates.cpu(),pv=output.outputs.values.cpu(),ov=output.outputs.valid.cpu();
      Index emitted=0,internal=0,external=0;
      const auto& g=profile.graph;
      for(Index row=0;row<5;++row)if(valid[row].item<bool>()) {
        const Index node=coordinates[row][1].item<Index>(),time=coordinates[row][2].item<Index>(),position=coordinates[row][3].item<Index>();
        const auto& n=g.nodes[node];
        for(Index j=g.outgoing_ports.offsets[node];j<g.outgoing_ports.offsets[node+1];++j) {
          const Index slot=j-g.outgoing_ports.offsets[node],phase=n.identity||n.emit_phases.empty()?-1:n.emit_phases[slot];
          if(phase==-2||(phase>=0&&time%n.emit_period!=phase))continue;
          require(emitted<count,"missing emitted slot");
          check_tensor(meta[emitted].narrow(0,0,5),at::tensor(std::vector<Index>{coordinates[row][0].item<Index>(),node,time,position,slot},at::kLong));
          auto expected=x[row];if(!n.identity&&n.emission=="slot_affine")expected=at::matmul(expected,f.model.nodes[node].extra.at("emit_w_"+std::to_string(slot)))+f.model.nodes[node].extra.at("emit_b_"+std::to_string(slot));
          check_tensor(values[emitted++],expected);
          const auto binding=g.outgoing_ports.bindings[j];const Index target=binding.kind?g.edges[binding.id].target:node;
          const Index arrival=time+(binding.kind?g.edges[binding.id].delay:0);
          auto expected_meta=at::tensor(std::vector<Index>{coordinates[row][0].item<Index>(),target,arrival,binding.kind,binding.id,position},at::kLong);
          if(binding.kind){require(on[internal].item<bool>(),"missing edge");check_tensor(edges[internal],expected_meta);check_tensor(ev[internal++],expected*f.model.edge_scale[binding.id]);}
          else {require(ov[external].item<bool>(),"missing output");check_tensor(ports[external],expected_meta);check_tensor(pv[external++],expected*f.model.output_scale[binding.id]);}
        }
      }
      require(emitted==count&&on.sum().item<Index>()==internal&&ov.sum().item<Index>()==external,"unexpected slot/physical delivery");
      require(emission.chunks().cpu().item<Index>()>0,"slot affine never executed");++cases;
    }
  }
  return cases;
}
Index windows(at::Device device) {
  Index cases=0;
  for(int shape:{0,1})for(int variant:{0,1})for(bool prefill:{false,true})for(Index chunk:{1,4}) {
    auto f=test::fixture(shape,variant);
    for(Index n=0;n<Index(f.graph.nodes.size());++n) {
      auto& node=f.graph.nodes[n];if(node.identity)continue;
      node.full=n%2?"lh-relu-identity-v1":"tanh";node.emission=n%2?"broadcast":"slot_affine";
      node.emit_period=3;
      const Index slots=f.graph.outgoing_ports.offsets[n+1]-f.graph.outgoing_ports.offsets[n];
      for(Index s=0;s<slots;++s)node.emit_phases.push_back(s%4==0?-1:s%4==1?-2:s%3);
    }
    configure(f,3);ContentLimits l;l.queue=128;l.arrivals=256;l.outputs=256;l.trace=2048;
    l.workspace_bytes=256*1024*1024;l.prefill=prefill;l.emission_chunk_rows=chunk;l.full_chunk_rows=chunk;
    ContentFlow flow(f.graph,f.model,f.initial,device,l);Streaming oracle(f.graph,f.model,{});Greedy greedy(f.graph,f.model,{});
    auto q=f.initial;Index previous=q.cut;
    for(Index offset:{2,6,11}) {
      const Index stop=f.initial.cut+offset;std::vector<External> input;
      for(const auto& x:f.input)if(x.time>=previous&&x.time<stop)input.push_back(x);
      auto expected=oracle.run(q,input,stop,stop);
      tide_bench::compare(greedy.run(q,input,stop,stop),expected,true,at::kFloat);
      tide_bench::compare(flow.advance(input,stop),expected,true,at::kFloat);q=expected.continuation;previous=stop;++cases;
    }
    l.prefill=!prefill;l.diagnostics=false;l.trace=0;
    ContentFlow switched(f.graph,f.model,flow.snapshot(),device,l);
    tide_bench::compare(switched.advance({},previous+3),oracle.run(q,{},previous+3,previous+3),false,at::kFloat);++cases;
  }
  for(bool prefill:{false,true}) {
    auto f=test::fixture(2,1);for(Index n=0;n<Index(f.graph.nodes.size());++n) {
      auto& node=f.graph.nodes[n];node.emission="slot_affine";
      node.emit_phases.assign(f.graph.outgoing_ports.offsets[n+1]-f.graph.outgoing_ports.offsets[n],-2);
    }
    configure(f,3);ContentLimits l;l.prefill=prefill;ContentFlow flow(f.graph,f.model,f.initial,device,l);
    Streaming oracle(f.graph,f.model,{});const auto stop=f.initial.cut+11;
    auto actual=flow.advance(f.input,stop);tide_bench::compare(actual,oracle.run(f.initial,f.input,stop,stop),true,at::kFloat);
    require(actual.outputs.empty()&&actual.messages.empty()&&!actual.trace.empty(),"all-absent emission erased computation or created messages");
    require(actual.stats.at("emission_chunks")==0,"all-absent slots performed affine work");++cases;
  }
  return cases;
}
Index refusals(at::Device device,at::ScalarType dtype) {
  Index cases=0;
  {auto f=component_fixture(3);test::model_dtype(f.model,dtype);f.graph.nodes[2].emit_phases={-2,-2};bool refused=false;
    try{ContentProfile invalid(f.graph,f.model,device);}catch(const std::invalid_argument&){refused=true;}
    require(refused,"identity boundary accepted conditional emission");++cases;}
  for(int kind=0;kind<5;++kind) {
    auto f=component_fixture(3);test::model_dtype(f.model,dtype);
    if(kind>=3){f.graph.nodes[0].emit_phases={-2,-2,kind==3?-1:-2,-2};configure(f,3);test::model_dtype(f.model,dtype);}
    ContentProfile profile(f.graph,f.model,device);
    PackedEmission emission(profile,device,1,kind==0?1:16,kind==1?1:16,1,64*1024*1024);
    const Index time=kind>=3?std::numeric_limits<Index>::max()-1:0;
    auto coords=at::tensor(std::vector<Index>{0,kind==2?99:0,time,time,0,0,time,time},at::kLong).reshape({2,4}).to(device);
    ActionBatch action{coords,at::ones({2,3},coords.options().dtype(dtype)),at::ones({2},coords.options().dtype(at::kBool))};
    auto error=at::zeros({1},coords.options().dtype(at::kInt));CannProgram program(device);
    auto out=emission.append_stage(program,action,error);program.finish();program.run();
    const auto expected=kind<2?1:kind==2?2:kind==3?3:0;
    require(error.cpu().item<int>()==expected,"wrong emission refusal/absent overflow behavior");
    require(!out.arrivals.valid.cpu().any().item<bool>()&&!out.outputs.valid.cpu().any().item<bool>(),"refusal/absent emission exposed deliveries");
    require(out.values.cpu().abs().sum().item<float>()==0,"failed metadata preflight performed numerical work");++cases;
  }
  return cases;
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);
    if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||(args.dtype!=at::kFloat&&args.dtype!=at::kHalf))throw std::invalid_argument("emission gate requires explicit NPU FP32/FP16");
    args.allow_npu_float16=true;
    auto device=portable_torch::resolve_device(args);if(device.type()!=c10::DeviceType::PrivateUse1)throw std::invalid_argument("emission gate requires NPU");
    at::set_num_threads(1);at::set_num_interop_threads(1);at::NoGradGuard guard;
    const auto a=components(device,args.dtype),b=args.dtype==at::kFloat?windows(device):0,c=refusals(device,args.dtype);
    std::cout<<"device-emission: passed components="<<a<<" windows="<<b<<" refusals="<<c<<" scope="<<(args.dtype==at::kFloat?"FP32_HARD_inference":"FP16_component_only")<<" keep_dtype=true\n";
    runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
