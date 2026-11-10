// Construction-only witness: a first local reference is not new owner storage.
// This preserves and attributes the current refusal; it does not relax budgets.
#include "weighted_row_graph.h"
#include "portable_torch/runtime.hpp"
#include <ATen/Parallel.h>
#include <iostream>
using namespace tide::device_online;
int main(int argc,char** argv){portable_torch::RuntimeSession runtime;
  try{
    auto args=portable_torch::parse_cli(argc,argv,true);
    if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)
      throw std::invalid_argument("weighted budget witness requires explicit accelerator FP32");
    const auto device=portable_torch::resolve_device(args);at::set_num_threads(1);at::set_num_interop_threads(1);at::NoGradGuard guard;
    constexpr int64_t rows=1,banks=64,width=32,available=32*1024;
    const auto opts=at::TensorOptions().device(device).dtype(at::kFloat);
    auto input=at::ones({rows,width},opts),owners=at::zeros({rows},opts.dtype(at::kLong)),error=at::zeros({1},opts.dtype(at::kInt));
    WeightedProjectionParameters parameters{at::zeros({banks,width,width},opts),{}};
    const std::vector<at::Tensor> held{input,owners,error,parameters.matrix};
    DeviceProgram program(device);program.limit_workspace(64*1024*1024);
    const auto local_before=program.retained_tensor_bytes(),owner_before=program.retained_tensor_bytes(held);
    bool rejected=false;std::string refusal;
    try{
      WeightedRowGraph graph(program,owners,banks,error,available);
      graph.projection(graph.input(input),parameters,"witness.matrix");
    }catch(const std::invalid_argument& e){
      refusal=e.what();rejected=refusal.find("local row graph tensor budget exceeded")!=std::string::npos
        &&refusal.find("kind=projection")!=std::string::npos;
    }
    const auto local_delta=program.retained_tensor_bytes()-local_before;
    const auto owner_delta=program.retained_tensor_bytes(held)-owner_before;
    if(!rejected||owner_delta<0||owner_delta>=available||local_delta<=available)
      throw std::runtime_error("borrowed storage witness did not isolate the local/global accounting boundary: "+refusal);
    const auto borrowed_first_use=local_delta-owner_delta;
    if(borrowed_first_use<parameters.matrix.storage().nbytes())
      throw std::runtime_error("borrowed parameter bank missing from accounting difference");
    program.close();
    std::cout<<"weighted-budget: reproduced_local_refusal=true allowance="<<available<<" caller_bank_bytes="<<parameters.matrix.storage().nbytes()
      <<" local_program_delta="<<local_delta<<" genuinely_new_owner_storage="<<owner_delta
      <<" borrowed_first_use_bytes="<<borrowed_first_use<<" message="<<refusal<<'\n';return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
