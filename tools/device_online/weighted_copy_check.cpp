// Static layout copies preserve exact bool/int64 metadata and float values.
// Regression for repeated-owner bool flag scatters in a complete window VJP.
#include "weighted_flow_internal.h"
#include "weighted_recurrent_internal.h"
#include "portable_torch/runtime.hpp"
#include <ATen/Parallel.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <iostream>
using namespace tide::device_online;
int main(int argc,char** argv){portable_torch::RuntimeSession runtime;
  try{auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("weighted copy check requires explicit accelerator FP32");
    auto device=portable_torch::resolve_device(args);at::set_num_threads(1);at::set_num_interop_threads(1);at::NoGradGuard guard;
    for(int owner=0;owner<18;++owner){
      auto opts=at::TensorOptions().dtype(at::kLong).device(device);
      auto flags=at::zeros({2,5},opts.dtype(at::kBool)),source=at::zeros({2,5},flags.options());
      auto times=at::zeros({32,6,2},opts),active=at::zeros(times.sizes(),flags.options());
      auto journal=at::zeros({6,32,2},opts),present=at::zeros(journal.sizes(),flags.options());
      auto raw=at::zeros({2,115},opts.dtype(at::kFloat)),values=at::zeros({2,3},raw.options());
      auto recurrent=at::zeros({2,4},flags.options());std::vector<at::Tensor> connected;
      DeviceProgram p(device);p.limit_workspace(64*1024*1024);
      for(int field=0;field<5;++field)weighted_copy(p,flags.select(1,field),source.select(1,field));
      for(int node=0;node<6;++node){weighted_copy(p,times.select(1,node),journal[node]);weighted_copy(p,active.select(1,node),present[node]);}
      weighted_copy(p,raw.narrow(1,7,3),values);
      for(int field=0;field<4;++field){connected.push_back(at::zeros({2,1},flags.options()));
        recurrent_detail::insert(p,recurrent,field,connected.back());}
      p.finish();
      for(int run=0;run<3;++run){
        auto want=at::arange(10,at::kLong).reshape({2,5}).add(owner+run).remainder(3).eq(0);
        auto j=at::arange(384,at::kLong).reshape({6,32,2}).add((int64_t(1)<<54)+owner*1000+run);
        auto v=at::arange(6,at::kFloat).reshape({2,3})*.17+run;
        source.copy_(want);journal.copy_(j);present.copy_(j.remainder(3).eq(0));values.copy_(v);
        for(int field=0;field<4;++field)connected[field].copy_(want.select(1,field).reshape({2,1}));
        flags.fill_(true);active.fill_(true);times.fill_(-1);raw.fill_(-.3);recurrent.fill_(true);
        c10::impl::VirtualGuardImpl(device.type()).synchronizeDevice(device.index());p.run(10000);
        auto expected=at::full({2,115},-.3,at::kFloat);expected.narrow(1,7,3).copy_(v);
        if(!at::equal(flags.cpu(),want)||!at::equal(times.cpu(),j.permute({1,0,2}))
            ||!at::equal(active.cpu(),j.remainder(3).eq(0).permute({1,0,2}))||!at::equal(raw.cpu(),expected)
            ||!at::equal(recurrent.cpu(),want.narrow(1,0,4)))
          throw std::runtime_error("weighted strided copy differs from independent CPU indexing");
      }
      p.close();
    }
    std::cout<<"weighted-copy: passed owners=18 submissions=54 exact_bool_int64=true strided_source_target=true recurrent_flags=true\n";return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
