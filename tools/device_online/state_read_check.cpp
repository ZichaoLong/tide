#include "device_backend.h"
#include "content_profile.h"
#include "content_fixture.h"
#include "portable_torch/runtime.hpp"
#include "tide/kernel.h"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using namespace tide;
using namespace tide::device_online;
constexpr Index nodes=6,samples=2,steps=4;
const Index epoch=(Index(1)<<55)/3*3+1,observations=(Index(1)<<55)+7;
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void same(const Tensor& a,const Tensor& b,const char* why) {
  const bool floating=b.is_floating_point();
  require(a.scalar_type()==b.scalar_type()&&(floating?at::allclose(a.cpu(),b,2e-5,2e-6):at::equal(a.cpu(),b)),why);
}
test::Fixture fixture(Index width,at::ScalarType dtype,const std::string& mode) {
  test::Fixture f;
  for(Index n=0;n<nodes;++n) {
    Node node;node.region=n;node.identity=n==0;node.memory=n%3==1?"ema":n%3==2?"lh-add-repeat-v1":"identity";
    node.full="identity";node.readout=n%2?"norm-fp32-v1":"linear-v1";
    if(!node.identity)node.state_clock={3,1,1};node.clear=mode=="content"&&n>=3;
    Region region;region.budget=1;region.observe_all=mode!="content"||n<3;region.read_mode=mode;
    f.graph.nodes.push_back(node);f.graph.regions.push_back(region);f.graph.outputs.push_back(n);
    auto v=at::arange(width,at::kFloat);
    NodeWeights w{at::zeros({width},at::kFloat),at::eye(width,at::kFloat),at::zeros({width},at::kFloat),v.remainder(7)*.03125f-.0625f};
    // decay zero gives an exactly representable sigmoid across CPU and device;
    // non-dyadic Add retention still exercises rounding at every repeated tick.
    if(n%3==2)w.extra["add_retention"]=at::full({},.99123f,at::kFloat);
    f.model.nodes.push_back(w);f.model.output_scale.push_back(at::ones({},at::kFloat));
  }
  // Identity boundaries have linear Read, regardless of the other nodes.
  f.graph.nodes[0].readout="linear-v1";f.graph.compile();test::model_dtype(f.model,dtype);
  return f;
}
struct Snapshot {Tensor state,clocks,present;};
Snapshot run(at::Device device,at::ScalarType dtype,Index width,const std::string& mode,
             bool vectorized,bool split,bool trace,Index& windows,Index& refusals) {
  auto f=fixture(width,dtype,mode);ContentProfile profile(f.graph,f.model,device);
  Model oracle=f.model;configure_model(f.graph,oracle);
  const Index capacity=samples*nodes*steps+2;
  auto longs=at::TensorOptions().device(device).dtype(at::kLong),floats=longs.dtype(dtype);
  auto initial=(at::arange(samples*nodes*width,at::kFloat).reshape({samples,nodes,width}).remainder(19)*.0137f-.093f).to(dtype);
  auto initial_clocks=at::zeros({samples,nodes,2},at::kLong);
  initial_clocks.select(2,0).fill_(epoch-3);initial_clocks.select(2,1).fill_(observations);
  ContentState state{initial.to(device),initial_clocks.to(device),at::ones({samples,nodes},longs.dtype(at::kBool))};
  auto host_fibers=at::full({capacity,4},-9,at::kLong);
  auto host_content=at::full({capacity,width},std::numeric_limits<float>::quiet_NaN(),at::TensorOptions().dtype(dtype));
  auto host_active=at::zeros({capacity},at::kBool);
  ReadyBatch ready;ready.fibers=host_fibers.to(device);ready.counts=at::zeros({3},longs);
  ContentBatch content{host_content.to(device),at::zeros({capacity},longs.dtype(at::kFloat)),{}};
  SelectionProposal selection;selection.active=host_active.to(device);selection.controls=at::zeros_like(content.scores);
  auto coefficients=at::empty_like(profile.decay),error=at::zeros({1},longs.dtype(at::kInt));
  auto stages=at::full({1},7,longs),total=at::zeros({1},longs);
  ContentLimits limits;limits.diagnostics=trace;limits.vectorized_state=vectorized;limits.max_repeat_ticks=16;
  DeviceProgram program(device);program.sigmoid(profile.decay,coefficients);
  append_read(program,profile,ready,content,state,coefficients,error,16,vectorized);
  auto output=append_content_state(program,profile,ready,content,selection,state,coefficients,stages,total,error,limits);
  commit_content_state(program,state,output,error);program.finish();
  std::vector<State> expected;
  for(Index b=0;b<samples;++b)for(Index n=0;n<nodes;++n)expected.push_back({initial[b][n],epoch-3,observations});
  const std::vector<Index> offsets{0,3,12,15};
  for(Index window=0;window<(split?steps:1);++window) {
    Index count=0;
    for(Index b=0;b<samples;++b)for(Index n=0;n<nodes;++n)for(Index s=split?window:0;s<(split?window+1:steps);++s,++count) {
      host_fibers[count].copy_(at::tensor({b,n,epoch+offsets[s],Index(0)},at::kLong));
      auto value=(at::arange(width,at::kFloat).remainder(11)*.00313f+.0117f*(1+s)-.0073f*(n+b)).to(dtype);
      if(s==1&&n==4)value.zero_();
      if(width==2048&&s==2&&n==1)value.fill_(4096); // Finite FP32 norm exceeds half range.
      host_content[count].copy_(value);host_active[count].fill_((b+n+s)%3!=0);
    }
    ready.fibers.copy_(host_fibers);ready.counts[1].fill_(count);content.content.copy_(host_content);
    selection.active.copy_(host_active);selection.controls.copy_(host_active.to(device,at::kFloat));
    // The candidate sees only independently constructed input tables, never
    // an oracle's event trace, proposal, score or parameter update.
    program.run();require(error.cpu().item<int>()==0,"state/Read valid input refused");
    require(content.scores.scalar_type()==at::kFloat&&output.event_values.scalar_type()==at::kFloat,"scoring/journal precision changed");
    auto actual=output.event_values.cpu(),metadata=output.event_meta.cpu();
    for(Index row=0;row<count;++row) {
      const Index b=host_fibers[row][0].item<Index>(),n=host_fibers[row][1].item<Index>(),time=host_fibers[row][2].item<Index>();
      const auto& spec=f.graph.nodes[n];const auto& region=f.graph.regions[n];const auto& w=oracle.nodes[n];
      const auto old=expected[b*nodes+n];ContentView view{host_content[row]};
      const auto proposed=w.kernel->step(w,old,view,time);
      const bool active=host_active[row].item<bool>();const auto comparison=region.observe_all||active?proposed:old;
      const auto next=spec.clear&&active?w.kernel->reset(comparison):comparison;
      const auto visible=(mode=="content"?view.value:mode=="old"?old.value:proposed.value).to(at::kFloat);
      const auto score=spec.identity?at::zeros({},at::kFloat):n%2?visible.norm():(visible*w.read.to(at::kFloat)).sum();
      same(content.scores[row],score,"FP32 Read score mismatch");
      same(output.comparison[row],comparison.value,"stored comparison mismatch");
      if(trace) {
        std::vector<Tensor> fields{view.value,old.value,proposed.value,comparison.value,next.value};
        for(Index field=0;field<5;++field)same(actual[row].narrow(0,field*width,width),fields[field].to(at::kFloat),"state field differs from CPU kernel");
        same(actual[row][5*width],score,"score journal mismatch");
        same(actual[row][5*width+1],at::full({},float(active),at::kFloat),"control journal mismatch");
        same(metadata[row],at::tensor({b,n,time,Index(active),old.last_time,old.observations,proposed.last_time,
          proposed.observations,comparison.last_time,comparison.observations,next.last_time,next.observations,Index(7)},at::kLong),"int64 metadata mismatch");
      }
      expected[b*nodes+n]=next;
    }
    for(Index b=0;b<samples;++b)for(Index n=0;n<nodes;++n) {
      const auto& e=expected[b*nodes+n];same(state.values[b][n],e.value,"continued state mismatch");
      same(state.clocks[b][n],at::tensor({e.last_time,e.observations},at::kLong),"continued clock mismatch");
    }
    ++windows;
  }
  require(total.cpu().item<Index>()==samples*nodes*steps,"event total mismatch");
  auto before=state.values.cpu();ready.counts.zero_();program.run();
  same(state.values,before,"empty replay changed state");++windows;
  // Failed transactions must never publish their partial state proposals.
  if(width==1&&!split&&trace) {
    auto preserved=state.clocks.cpu();ready.counts[1].fill_(1);error.fill_(6);program.run();
    require(error.cpu().item<int>()==6,"sticky error changed");same(state.values,before,"sticky error committed state");++refusals;
    error.zero_();ready.fibers[0][0].fill_(samples);program.run();
    require(error.cpu().item<int>()==2,"invalid owner accepted");same(state.values,before,"invalid owner committed state");
    same(state.clocks,preserved,"invalid owner committed clock");++refusals;
    error.zero_();ready.fibers[0][0].fill_(0);ready.fibers[0][1].fill_(2);ready.fibers[0][2].fill_(epoch+18+300);
    program.run();require(error.cpu().item<int>()==8,"repeat work bound missing");same(state.values,before,"bounded refusal committed state");++refusals;
  }
  return {state.values.cpu(),state.clocks.cpu(),state.present.cpu()};
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    bool smoke=false;std::vector<char*> argsv{argv[0]};
    for(int i=1;i<argc;++i)if(std::string(argv[i])=="--profile-smoke")smoke=true;else argsv.push_back(argv[i]);
    auto args=portable_torch::parse_cli(argsv.size(),argsv.data(),true);
    if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||(args.dtype!=at::kFloat&&args.dtype!=at::kHalf))throw std::invalid_argument("state/Read requires explicit NPU FP32/FP16");
    args.allow_npu_float16=true;auto device=portable_torch::resolve_device(args);
    if(device.type()!=tide::device_online::resident_device_type)throw std::invalid_argument("state/Read requires NPU");
    at::set_num_threads(1);at::set_num_interop_threads(1);at::NoGradGuard guard;
    Index cases=0,windows=0,refusals=0;
    for(Index width:smoke?std::vector<Index>{257}:std::vector<Index>{1,7,257,2048})
    for(const std::string mode:smoke?std::vector<std::string>{"proposal"}:std::vector<std::string>{"content","old","proposal"})
    for(bool vectorized:{false,true})for(bool trace:{false,true}) {
      if(smoke&&(!vectorized||!trace))continue;
      auto batch=run(device,args.dtype,width,mode,vectorized,false,trace,windows,refusals);
      auto step=run(device,args.dtype,width,mode,vectorized,true,trace,windows,refusals);
      same(batch.state,step.state,"node-time batch changed per-event rounding");
      same(batch.clocks,step.clocks,"node-time batch changed clocks");
      same(batch.present,step.present,"node-time batch changed presence");++cases;
    }
    std::cout<<"state-read: passed configurations="<<cases<<" windows="<<windows<<" refusals="<<refusals
      <<" CPU=storage_dtype_canonical_kernels scoring=FP32 per_event_rounding=true scope="<<(smoke?"profile-smoke":"component-only")<<'\n';
    runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
