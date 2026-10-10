#include "device_backend.h"
#include "packed_aggregate.h"
#include "content_fixture.h"
#include "portable_torch/runtime.hpp"
#include "tide/aggregate.h"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using namespace tide;
using namespace tide::device_online;
constexpr Index capacity=32,groups=6;
const std::array<std::string,5> names{"sum","mean","weighted_mean","active_softmax","all_softmax"};
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
test::Fixture fixture(Index width,int kind,at::ScalarType dtype) {
  test::Fixture f;f.graph.nodes={{0}};f.graph.regions={{1}};f.graph.inputs={0,0,0,0};f.graph.outputs={0};
  auto& n=f.graph.nodes[0];n.memory="identity";n.full="identity";n.aggregation=names[kind];f.graph.compile();
  f.graph.source_domain->input={2,0,1,0};f.graph.compile();
  NodeWeights w{at::zeros({width},at::kFloat),at::eye(width,at::kFloat),at::zeros({width},at::kFloat),at::zeros({width},at::kFloat)};
  if(kind>=2)for(Index s=0;s<3;++s)w.extra[(kind==2?"agg_mass_":"agg_logit_")+std::to_string(s)]=at::full({},-.31f+.13f*s,at::kFloat);
  f.model.nodes={w};f.model.input_scale={at::full({},.3f,at::kFloat),at::full({},-1.1f,at::kFloat),
    at::full({},.7f,at::kFloat),at::full({},.53f,at::kFloat)};f.model.output_scale={at::ones({},at::kFloat)};
  test::model_dtype(f.model,dtype);return f;
}
struct Inputs {ReadyBatch ready;std::vector<AggregateInput> reference;Index atoms=0;};
Inputs inputs(const ContentProfile& p,Index width,int variant) {
  Inputs r;auto& h=r.ready;
  h.atoms.coordinates=at::full({capacity,6},-9,at::kLong);
  h.atoms.values=at::full({capacity,width},std::numeric_limits<float>::quiet_NaN(),at::TensorOptions().dtype(p.dtype));
  h.fibers=at::full({capacity,4},-9,at::kLong);h.fiber_offsets=at::zeros({capacity+1},at::kLong);
  for(Index g=0;g<groups;++g) {
    const Index time=(Index(1)<<55)+g;h.fibers[g].copy_(at::tensor({Index(0),Index(0),time,g},at::kLong));
    AggregateInput query{time,3,{}};
    for(Index port=0;port<4;++port) {
      if(port==0&&g%2==0||port==1&&variant||port==3&&!variant)continue;
      auto value=(at::arange(width,at::kFloat).remainder(13)*.0237f-.101f+.0513f*g+.013f*port).to(p.dtype);
      if(port==0&&g==1)value.zero_(); // Present zero still belongs to the domain.
      if(variant)value=-value;
      h.atoms.coordinates[r.atoms].copy_(at::tensor({Index(0),Index(0),time,Index(0),port,g},at::kLong));
      h.atoms.values[r.atoms++].copy_(value);
      query.sources.push_back({p.graph.source_domain->input[port],{0,0,time,0,port,g,value},p.model.input_scale[port]});
    }
    h.fiber_offsets[g+1].fill_(r.atoms);r.reference.push_back(query);
  }
  h.counts=at::tensor({r.atoms,groups,groups},at::kLong);return r;
}
void check(at::Device device,at::ScalarType dtype,Index width,int kind,bool vectorized,Index chunk,Index& replays) {
  auto f=fixture(width,kind,dtype);ContentProfile profile(f.graph,f.model,device);auto data=inputs(profile,width,0);
  ReadyBatch ready;ready.atoms.coordinates=data.ready.atoms.coordinates.to(device);ready.atoms.values=data.ready.atoms.values.to(device);
  ready.fibers=data.ready.fibers.to(device);ready.fiber_offsets=data.ready.fiber_offsets.to(device);ready.counts=data.ready.counts.to(device);
  auto error=at::zeros({1},ready.counts.options().dtype(at::kInt));
  PackedAggregate aggregate(profile,device,capacity,chunk,16*1024*1024);DeviceProgram program(device);
  auto out=append_content(program,profile,ready,error,vectorized,&aggregate);program.finish();
  auto kernel=make_aggregate_kernel(f.graph.nodes[0]);
  for(int variant:{0,1}) {
    data=inputs(profile,width,variant);ready.atoms.coordinates.copy_(data.ready.atoms.coordinates);
    ready.atoms.values.copy_(data.ready.atoms.values);ready.fiber_offsets.copy_(data.ready.fiber_offsets);ready.counts.copy_(data.ready.counts);
    out.content.fill_(std::numeric_limits<float>::quiet_NaN());out.weighted.fill_(std::numeric_limits<float>::quiet_NaN());
    program.run();require(error.cpu().item<int>()==0,"normalized payload refused valid domain");
    auto actual=out.content.cpu(),weighted=out.weighted.cpu();require(actual.scalar_type()==dtype&&weighted.scalar_type()==dtype,"Aggregate changed payload dtype");
    Index atom=0;
    for(Index g=0;g<groups;++g) {
      // Canonical CPU Aggregate consumes the stored physical contributions in
      // FP32. Half rounding of physical products precedes normalization.
      auto query=data.reference[g];auto w=f.model.nodes[0];
      for(auto& [_,x]:w.extra)x=x.to(at::kFloat);
      for(auto& source:query.sources) {
        source.atom.value=(source.atom.value.to(at::kFloat)*source.scale.to(at::kFloat)).to(dtype).to(at::kFloat);
        source.scale=at::ones({},at::kFloat);
      }
      const auto expected=kernel->step(w,query);
      const auto rtol=dtype==at::kHalf?2e-3:1e-5,atol=dtype==at::kHalf?2e-4:1e-6;
      require(at::allclose(actual[g],expected.value.to(dtype),rtol,atol),"normalized content differs from canonical CPU Aggregate");
      // Public contributions are sorted by logical slot. Device storage keeps
      // physical source order; match by identity instead of positional zip.
      for(const auto& source:query.sources) {
        const auto it=std::find_if(expected.contributions.begin(),expected.contributions.end(),
          [&](const auto& contribution){return contribution.slot==source.slot;});
        require(it!=expected.contributions.end(),"missing logical contribution");
        require(at::allclose(weighted[atom++],it->value.to(dtype),rtol,atol),"normalized source contribution differs");
      }
      auto ideal=data.reference[g];for(auto& source:ideal.sources){source.atom.value=source.atom.value.to(at::kDouble);source.scale=source.scale.to(at::kDouble);}
      for(auto& [_,x]:w.extra)x=x.to(at::kDouble);
      auto expected64=kernel->step(w,ideal);
      require(at::allclose(actual[g].to(at::kDouble),expected64.value,dtype==at::kHalf?4e-3:2e-5,dtype==at::kHalf?3e-4:2e-6),"normalized content exceeds FP64 error budget");
    }
    require(atom==data.atoms,"physical contribution count changed");
    require(actual.narrow(0,groups,capacity-groups).isnan().all().item<bool>(),"normalization wrote absent fibers");
    require(weighted.narrow(0,atom,capacity-atom).isnan().all().item<bool>(),"normalization wrote absent atoms");++replays;
  }
  ready.counts.zero_();ready.fiber_offsets.zero_();program.run();require(error.cpu().item<int>()==0,"empty normalization failed");++replays;
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);
    if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||(args.dtype!=at::kFloat&&args.dtype!=at::kHalf))throw std::invalid_argument("Aggregate payload gate requires NPU FP32/FP16");
    args.allow_npu_float16=true;auto device=portable_torch::resolve_device(args);
    if(device.type()!=tide::device_online::resident_device_type)throw std::invalid_argument("Aggregate payload gate requires NPU");
    at::set_num_threads(1);at::set_num_interop_threads(1);at::NoGradGuard guard;
    Index cases=0,replays=0;
    for(Index width:{1,33,257,2048})for(int kind:{1,2,3,4})for(bool vectorized:{false,true})for(Index chunk:{1,4}) {
      check(device,args.dtype,width,kind,vectorized,chunk,replays);++cases;
    }
    std::cout<<"aggregate-payload: passed configurations="<<cases<<" replays="<<replays<<" CPU=FP32_FP64 normalization=FP32 scope=component_only\n";
    runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
