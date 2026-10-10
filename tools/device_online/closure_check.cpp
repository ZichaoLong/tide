#include "device_backend.h"
#include "device_program.h"
#include "queue_closure.h"
#include "portable_torch/runtime.hpp"
#include "device_launch_tide_closure.h"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <iostream>
#include <limits>

namespace {
using namespace tide::device_online;
using I=int64_t;
uint8_t* address(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}
void check(at::Device device) {
  validate_kernel_device(device);
  at::NoGradGuard guard;
  auto longs=at::TensorOptions().device(device).dtype(at::kLong),ints=longs.dtype(at::kInt);
  const I capacity=16,nodes=4,regions=3,samples=2;
  const std::vector<I> owners{0,1,1,2};
  const std::vector<std::vector<Wire>> graphs{
    {}, {{0,1,1},{2,3,3},{0,1,1}}, {{0,1,2},{1,0,3},{2,3,1},{3,2,2}},
    {{0,0,1},{0,2,3},{2,1,2},{3,0,1}},
    {{0,1,std::numeric_limits<I>::max()},{1,3,2},{3,0,3}}};
  I cases=0;
  for(const auto& wires:graphs)for(bool prefill:{false,true}) {
    QueueClosure cpu(owners,regions,wires,samples,at::Device(at::kCPU));
    QueueClosure metadata(owners,regions,wires,samples,device);
    auto coords=at::zeros({capacity,6},longs),valid=at::zeros({capacity},longs.dtype(at::kBool));
    auto stop=at::ones({1},longs),ready=at::zeros({capacity},ints),go=at::zeros({1},ints);
    auto error=at::zeros({1},ints),workspace=at::zeros({samples*regions*2+samples},longs);
    auto calls=at::zeros({1},longs),one=at::ones({1},longs),again=at::zeros({1},ints);
    auto ownership=metadata.owners(),distances=metadata.distances();
    portable_torch::synchronize(device);
    DeviceProgram program(device);
    auto active=program.label(),end=program.label();
    program.kernel([=](void* stream) {
      check_device_launch(TIDE_LAUNCH_KERNEL(tide_closure)(1,stream,address(coords),address(valid),
        address(ownership),address(distances),address(workspace),address(stop),address(ready),
        address(go),address(error),capacity,nodes,regions,samples,I(prefill)),"launch Ascend C closure");
    },{coords,valid,ownership,distances,workspace,stop,ready,go,error});
    program.branch(go,{end,active});program.mark(active);program.add(calls,one);
    program.branch(again,{end});program.mark(end);program.finish();
    for(int variant=0;variant<3;++variant) {
      I base=variant==1?(I(1)<<55)+17:variant==2?std::numeric_limits<I>::max()-10:0;
      std::vector<I> rows;
      for(I b=0;b<samples;++b)for(I n=0;n<nodes;++n)for(I j=0;j<2;++j)
        rows.insert(rows.end(),{b,n,base+(n*3+j*5+b+variant)%9,0,n,j});
      auto input=at::tensor(rows,at::kLong).reshape({capacity,6});
      auto present=at::arange(capacity,at::kLong).remainder(3)!=variant;
      auto cpu_stop=at::full({1},base+8,at::kLong);
      coords.copy_(input);valid.copy_(present);stop.copy_(cpu_stop);calls.zero_();
      portable_torch::synchronize(device);program.run();
      auto expected=cpu.ready({input,at::zeros({capacity,1}),present},cpu_stop,prefill);
      if(error.cpu().item<int>()!=0||!at::equal(ready.cpu().to(at::kBool),expected)
          ||calls.cpu().item<I>()!=I(expected.any().item<bool>()))
        throw std::runtime_error("Ascend C exact-int64 closure/device branch mismatch");
      ++cases;
    }
    valid.zero_();calls.zero_();portable_torch::synchronize(device);program.run();
    if(error.cpu().item<int>()!=0||ready.cpu().any().item<bool>()||calls.cpu().item<I>()!=0)
      throw std::runtime_error("empty device queue executed an active branch");
    ++cases;
    coords.zero_();valid.fill_(true);stop.zero_();calls.zero_();
    portable_torch::synchronize(device);program.run();
    if(error.cpu().item<int>()!=0||ready.cpu().any().item<bool>()||calls.cpu().item<I>()!=0)
      throw std::runtime_error("right-boundary messages executed before the next window");
    ++cases;
    // Invalid live indices are refused on-device before indexing topology.
    coords.zero_();valid.fill_(true);coords.select(1,1).fill_(nodes);calls.zero_();
    portable_torch::synchronize(device);program.run();
    if(error.cpu().item<int>()!=2||ready.cpu().any().item<bool>()||calls.cpu().item<I>()!=0)
      throw std::runtime_error("Ascend C closure accepted invalid device coordinates");
    ++cases;program.close();
  }
  std::cout<<"device-closure: passed cases="<<cases<<" exact_int64=true host_decisions_per_call=0"
           <<" cpu_fallback=false scope=queue_readiness_and_branch_only\n";
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);
    if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("closure check requires explicit NPU and float32");
    auto device=portable_torch::resolve_device(args);
    if(device.type()!=tide::device_online::resident_device_type)throw std::invalid_argument("closure check requires NPU");
    at::set_num_threads(1);at::set_num_interop_threads(1);check(device);return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
