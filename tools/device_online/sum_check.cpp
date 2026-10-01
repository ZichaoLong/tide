#include "packed_sum.h"
#include "portable_torch/runtime.hpp"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using namespace tide::device_online;
using I=int64_t;
constexpr I capacity=257,nodes=4,inputs=64,edges=448;
const float poison=std::numeric_limits<float>::quiet_NaN();
void require(bool yes,const char* why){if(!yes)throw std::runtime_error(why);}
struct Case {
  ReadyBatch host,device;
  at::Tensor sources,scales,device_sources,device_scales;
  I atoms,groups,width;
  Case(at::Device d,I w,const std::vector<I>& sizes,at::ScalarType dtype):atoms(0),groups(sizes.size()),width(w) {
    host.atoms={at::full({capacity,6},-999,at::kLong),at::full({capacity,w},poison,at::kFloat),{}};
    host.fiber_offsets=at::zeros({capacity+1},at::kLong);host.fibers=at::full({capacity,4},-999,at::kLong);
    sources=at::empty({inputs+edges,2},at::kLong);scales=at::full({inputs+edges},poison,at::kFloat);
    auto src=sources.accessor<I,2>();for(I k=0;k<inputs+edges;++k){src[k][0]=k%nodes;src[k][1]=k/nodes;}
    auto c=host.atoms.coordinates.accessor<I,2>(),f=host.fibers.accessor<I,2>();auto x=host.atoms.values.accessor<float,2>();
    for(I i=0;i<groups;++i) {
      I n=i%nodes;f[i][0]=i/nodes;f[i][1]=n;f[i][2]=11+i;f[i][3]=i;
      for(I j=0;j<sizes[i];++j,++atoms) {
        const I key=n+nodes*j;scales[key].fill_(j%3==0?0.f:(float(j%7)-3)/8.f);
        c[atoms][0]=i/nodes;c[atoms][1]=n;c[atoms][2]=11+i;
        c[atoms][3]=key<inputs?0:1;c[atoms][4]=key<inputs?key:key-inputs;c[atoms][5]=j;
        for(I col=0;col<w;++col)x[atoms][col]=(float((atoms*17+col*3)%71)-35)/32.f;
      }
      host.fiber_offsets[i+1].fill_(atoms);
    }
    host.counts=at::tensor({atoms,groups,groups},at::kLong);
    host.atoms.values=host.atoms.values.to(dtype);scales=scales.to(dtype);
    device.atoms.coordinates=host.atoms.coordinates.to(d);device.atoms.values=host.atoms.values.to(d);
    device.fibers=host.fibers.to(d);device.fiber_offsets=host.fiber_offsets.to(d);device.counts=host.counts.to(d);
    device_sources=sources.to(d);device_scales=scales.to(d);
  }
  PackedSum expected() const {
    PackedSum result{at::zeros({groups,width},host.atoms.values.options()),at::zeros({atoms,width},host.atoms.values.options())};
    const auto c=host.atoms.coordinates.accessor<I,2>();
    for(I i=0;i<groups;++i) {
      const I start=host.fiber_offsets[i].item<I>(),end=host.fiber_offsets[i+1].item<I>();
      at::Tensor total;
      for(I a=start;a<end;++a) {
        const I key=c[a][4]+(c[a][3]?inputs:0);
        auto contribution=host.atoms.values[a].to(at::kFloat)*scales[key].to(at::kFloat);result.weighted[a].copy_(contribution);
        total=total.defined()?total+contribution:contribution;
      }
      result.content[i].copy_(total);
    }
    return result;
  }
};
void compare(const PackedSum& out,const Case& c) {
  auto h=out.content.cpu(),z=out.weighted.cpu();auto expected=c.expected();
  require(h.scalar_type()==c.host.atoms.values.scalar_type()&&z.scalar_type()==h.scalar_type(),"sum changed payload precision");
  const bool half=h.scalar_type()==at::kHalf;
  require(at::allclose(h.narrow(0,0,c.groups),expected.content,half?1e-3:1e-5,half?1e-4:1e-6),"packed sum differs from CPU");
  require(at::allclose(z.narrow(0,0,c.atoms),expected.weighted,half?1e-3:1e-5,half?1e-4:1e-6),"packed contributions differ from CPU");
  require(h.narrow(0,c.groups,capacity-c.groups).isnan().all().item<bool>(),"sum overwrote absent fibers");
  require(z.narrow(0,c.atoms,capacity-c.atoms).isnan().all().item<bool>(),"sum overwrote absent atoms");
}
void check(at::Device d,at::ScalarType dtype) {
  at::NoGradGuard no_grad;I cases=0,replays=0,refusals=0;
  for(I width:{1,7,33,256,257,511,512,1025,2048})
  for(const auto& sizes:std::vector<std::vector<I>>{{},{1},{1,7,2,17,3},{64,3,5}}) {
    Case c(d,width,sizes,dtype);
    for(bool vectorized:{false,true}) {
      auto error=at::zeros({1},c.device.counts.options().dtype(at::kInt));CannProgram program(d);
      auto out=append_packed_sum(program,c.device,c.device_sources,c.device_scales,nodes,inputs,edges,error,vectorized);
      program.finish();
      for(I replay=0;replay<2;++replay) {
        out.content.fill_(poison);out.weighted.fill_(poison);
        c.host.atoms.values.narrow(0,0,c.atoms).mul_(-1);
        c.device.atoms.values.copy_(c.host.atoms.values);
        portable_torch::synchronize(d);program.run();
        require(error.cpu().item<int>()==0,"packed sum unexpectedly refused");compare(out,c);++replays;
      }
      ++cases;
    }
  }
  // The entire metadata preflight must precede numerical writes, including
  // failures found in a later group and sticky errors inherited from a stage.
  for(bool vectorized:{false,true})for(int failure=0;failure<9;++failure) {
    Case c(d,33,{3,2},dtype);auto error=at::zeros({1},c.device.counts.options().dtype(at::kInt));
    if(failure==0)c.device.counts[1].fill_(-1);
    if(failure==1)c.device.counts[1].fill_(capacity+1);
    if(failure==2)c.device.counts[0].fill_(capacity+1);
    if(failure==3)c.device.fiber_offsets[0].fill_(1);
    if(failure==4)c.device.fiber_offsets[2].fill_(capacity+1);
    if(failure==5)c.device_sources[5][1].copy_(c.device_sources[1][1]); // duplicate logical source in group2.
    if(failure==6)c.device.atoms.coordinates[4][3].fill_(2);
    if(failure==7)c.device.atoms.coordinates[4][2].fill_(999);
    if(failure==8)error.fill_(9);
    CannProgram program(d);auto out=append_packed_sum(program,c.device,c.device_sources,c.device_scales,nodes,inputs,edges,error,vectorized);
    program.finish();out.content.fill_(poison);out.weighted.fill_(poison);portable_torch::synchronize(d);program.run();
    require(error.cpu().item<int>()==(failure==8?9:2),"sum metadata refusal lost");
    require(out.content.cpu().isnan().all().item<bool>()&&out.weighted.cpu().isnan().all().item<bool>(),"refused sum executed payload arithmetic");
    ++refusals;
  }
  {
    Case c(d,7,{1},dtype);auto error=at::zeros({1},c.device.counts.options().dtype(at::kInt));CannProgram program(d);
    at::AutoGradMode enabled(true);bool refused=false;
    try{append_packed_sum(program,c.device,c.device_sources,c.device_scales,nodes,inputs,edges,error,true);}
    catch(const std::invalid_argument&){refused=true;}
    require(refused,"inference sum silently accepted autograd");
  }
  std::cout<<"packed-sum: passed cases="<<cases<<" replays="<<replays<<" refusals="<<refusals
    <<" tail_widths=true inactive_nan=true ordered_reduction=true accumulator=FP32 autograd_refused=true\n";
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);
    if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||(args.dtype!=at::kFloat&&args.dtype!=at::kHalf))throw std::invalid_argument("sum check requires explicit NPU FP32/FP16");
    args.allow_npu_float16=true;
    auto d=portable_torch::resolve_device(args);
    if(d.type()!=c10::DeviceType::PrivateUse1)throw std::invalid_argument("sum check requires NPU");
    at::set_num_threads(1);at::set_num_interop_threads(1);check(d,args.dtype);runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
