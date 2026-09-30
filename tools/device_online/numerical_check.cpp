#include "cann_program.h"
#include "portable_torch/runtime.hpp"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <iostream>

namespace {
void check(at::Device device,at::ScalarType dtype) {
  using tide::device_online::CannProgram;
  at::NoGradGuard no_grad;
  auto longs=at::TensorOptions().device(device).dtype(at::kLong), floats=longs.dtype(dtype);
  auto count=at::zeros({1},longs), limit=at::ones({1},longs), step=at::ones({1},longs);
  auto predicate=at::zeros({1},longs.dtype(at::kBool)), index=at::zeros({1},longs.dtype(at::kInt));
  auto again=at::zeros_like(index), values=at::ones({4,3},floats), gathered=at::empty_like(values);
  auto scale=at::full({1},0.5,floats), bias=at::ones({1},floats);
  auto order_cpu=at::tensor({2,0,3,1},at::kLong), order=order_cpu.to(device);
  portable_torch::synchronize(device);
  CannProgram program(device);
  auto head=program.label(), body=program.label(), end=program.label();
  program.mark(head);program.less(count,limit,predicate);program.cast_index(predicate,index);
  program.branch(index,{end,body});
  program.mark(body);
  program.index_select(values,0,order,gathered);program.multiply(gathered,scale,values);
  program.add(values,bias);program.add(count,step);program.branch(again,{head});
  program.mark(end);program.finish();
  for(auto loops:{3,1,0,4}) {
    auto input=(at::arange(12,at::kFloat).reshape({4,3})+loops).to(dtype);
    auto expected=input.clone();
    for(int i=0;i<loops;++i)expected=expected.index_select(0,order_cpu)*0.5+1.0;
    values.copy_(input);count.zero_();limit.fill_(loops);portable_torch::synchronize(device);
    program.run();
    if(!at::equal(values.cpu(),expected)||count.cpu().item<int64_t>()!=loops)
      throw std::runtime_error("device-controlled packed arithmetic differs from scalar CPU recurrence");
  }
  program.close();
  std::cout<<"device-numerical: passed cases=4 host_decisions_per_iteration=0 packed_rows=4\n";
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);
    if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||(args.dtype!=at::kFloat&&args.dtype!=at::kHalf))
      throw std::invalid_argument("numerical check requires explicit NPU and float32/float16");
    args.allow_npu_float16=true;
    auto device=portable_torch::resolve_device(args);
    at::set_num_threads(1);at::set_num_interop_threads(1);
    check(device,args.dtype);return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
