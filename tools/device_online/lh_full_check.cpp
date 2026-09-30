#include "packed_lh_full.h"
#include "content_fixture.h"
#include "lh_precision_check.h"
#include "portable_torch/runtime.hpp"
#include "tide/lh_full.h"
#include "tide/stream.h"
#include "tide/greedy.h"
#include "../../cpp/bench/streaming.h"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <iostream>
#include <limits>

namespace {
using namespace tide;
using namespace tide::device_online;
void require(bool yes,const char* why){if(!yes)throw std::runtime_error(why);}
std::vector<std::string> profiles() {
  std::vector<std::string> result;
  for(const std::string a:{"relu","silu","identity"})for(const std::string n:{"identity","rms","layer"})
    result.push_back("lh-"+a+"-"+n+"-v1");
  return result;
}
Index component(at::Device device,test::LhPrecision& precision) {
  Index cases=0;const Index nodes=11,rows=35;const auto names=profiles();
  const auto opts=at::TensorOptions().device(device).dtype(at::kFloat);
  std::vector<Index> kinds{1,2,3,4,5,6,7,8,9,0,2};
  for(Index width:{1,7,33,257})for(Index chunk:{1,4}) {
    auto weight=at::arange(nodes*width,at::kFloat).reshape({nodes,width}).remainder(5)/8+.75;
    auto bias=at::arange(nodes*width,at::kFloat).reshape({nodes,width}).remainder(3)/16;
    weight[10].fill_(std::numeric_limits<float>::quiet_NaN());bias[10].fill_(std::numeric_limits<float>::quiet_NaN());
    PackedLhFull full(kinds,weight,bias,device,rows,chunk,16*1024*1024);
    ActionBatch input{at::zeros({rows,4},opts.dtype(at::kLong)),at::zeros({rows,width},opts),at::zeros({rows},opts.dtype(at::kBool))};
    auto comparison=at::zeros_like(input.values),error=at::zeros({1},opts.dtype(at::kInt)),chunks=at::zeros({1},opts.dtype(at::kLong));
    CannProgram program(device);auto result=full.append_stage(program,input,comparison,error,chunks);program.finish();
    for(Index round=0;round<5;++round) {
      auto coords=at::zeros({rows,4},at::kLong),mask=at::zeros({rows},at::kBool);
      auto h=at::arange(rows*width,at::kFloat).reshape({rows,width})/128,state=at::sin(h)*.5;
      if(round==3)state.fill_(.25); // zero variance, including width one
      if(round==4)state=at::sin(h*128)*.5; // well-conditioned mixed-sign rows
      auto expected=h.clone();std::vector<Index> totals(10);
      for(Index i=0;i<rows;++i) {
        const Index n=i%nodes;const bool active=round!=0&&n!=10&&(round!=2||i%2==0);
        mask[i].fill_(active);coords[i][1].fill_(active?n:-77);coords[i][2].fill_((Index(1)<<55)+i);
        if(active&&kinds[n]) {
          NodeWeights w;w.bias=at::zeros({width},at::kFloat);w.full_kind=names[kinds[n]-1];
          w.extra={{"lh_norm_weight",weight[n]},{"lh_norm_bias",bias[n]}};
          expected[i].copy_(lh_full_fresh(w,state[i]));++totals[kinds[n]];
        }
        if(!active){state[i].fill_(std::numeric_limits<float>::quiet_NaN());h[i].fill_(std::numeric_limits<float>::quiet_NaN());}
      }
      input.coordinates.copy_(coords);input.values.copy_(h);input.valid.copy_(mask);comparison.copy_(state);chunks.zero_();
      portable_torch::synchronize(device);program.run();
      Index wanted_chunks=0;for(Index total:totals)wanted_chunks+=(total+chunk-1)/chunk;
      require(error.cpu().item<int>()==0,"LH Full refused valid selected rows");
      require(chunks.cpu().item<Index>()==wanted_chunks,"LH Full actual chunk count");
      const auto actual=result.values.cpu();
      for(Index i=0;i<rows;++i)if(mask[i].item<bool>()) {
        const Index node=i%nodes;
        if(kinds[node])precision.check(kinds[node],state[i],weight[node],bias[node],expected[i],actual[i]);
        else require(at::equal(actual[i],expected[i]),"non-LH Full changed");
      }
      if(round==4)require(at::allclose(actual.index({mask}),expected.index({mask}),1e-5,1e-6),"well-conditioned LH Full parity");
      ++cases;
    }
    program.close();
  }
  return cases;
}
Index windows(at::Device device) {
  const auto names=profiles();Index cases=0;
  for(int shape:{0,1})for(int variant:{0,1})for(bool prefill:{false,true})for(Index mode=0;mode<3;++mode) {
    auto f=test::fixture(shape,variant);
    for(size_t n=0;n<f.graph.nodes.size();++n)if(!f.graph.nodes[n].identity) {
      f.graph.nodes[n].full=names[(n+3*variant+mode)%9];f.graph.nodes[n].readout="norm-fp32-v1";
      f.model.nodes[n].extra["lh_norm_weight"]=at::tensor({.75f,1.f,1.25f});
      f.model.nodes[n].extra["lh_norm_bias"]=at::tensor({.03125f,-.015625f,0.f});
    }
    for(auto& r:f.graph.regions)r.read_mode=mode==0?"content":mode==1?"old":"proposal";
    f.graph.compile();f.initial.identity=f.graph.identity;
    ContentLimits l;l.queue=96;l.arrivals=128;l.outputs=256;l.trace=1024;l.prefill=prefill;l.full_chunk_rows=3;
    l.workspace_bytes=512*1024*1024;
    ContentFlow flow(f.graph,f.model,f.initial,device,l);Streaming oracle(f.graph,f.model,{});Greedy greedy(f.graph,f.model,{});
    auto q=f.initial;Index previous=q.cut;
    for(Index step:{2,6,11}) {
      const auto stop=f.initial.cut+step;std::vector<External> xs;
      for(const auto& x:f.input)if(x.time>=previous&&x.time<stop)xs.push_back(x);
      auto expected=oracle.run(q,xs,stop,stop);tide_bench::compare(greedy.run(q,xs,stop,stop),expected,true,at::kFloat);
      tide_bench::compare(flow.advance(xs,stop),expected,true,at::kFloat);q=expected.continuation;previous=stop;++cases;
    }
    l.prefill=!prefill;ContentFlow restored(f.graph,f.model,flow.snapshot(),device,l);
    tide_bench::compare(restored.advance({},previous+3),oracle.run(q,{},previous+3,previous+3),true,at::kFloat);++cases;
  }
  return cases;
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);
    if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("LH Full gate requires explicit NPU FP32");
    auto device=portable_torch::resolve_device(args);
    if(device.type()!=c10::DeviceType::PrivateUse1)throw std::invalid_argument("LH Full gate requires NPU");
    at::set_num_threads(1);at::set_num_interop_threads(1);at::NoGradGuard guard;
    test::LhPrecision precision;const auto cells=component(device,precision),count=windows(device);
    std::cout<<"device-lh-full: passed components="<<cells<<" windows="<<count<<" profiles=9 scope=FP32_broadcast_inference"
      <<" norm_rows="<<precision.normalized_rows<<" strict_component_misses="<<precision.strict_misses
      <<" cpu_fp64_max_abs="<<precision.max_cpu_error<<" device_fp64_max_abs="<<precision.max_device_error
      <<" max_condition_budget_fraction="<<precision.max_budget_fraction<<'\n';
    runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
