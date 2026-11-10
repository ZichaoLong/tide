// Construction-only witness: borrowed storage remains in the owner union but
// cannot exhaust a local allowance a second time. Real new storage still can.
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
    // Aliased views must neither duplicate owner cost nor add repeated credit.
    const std::vector<at::Tensor> held{input,owners,error,parameters.matrix,parameters.matrix[0],parameters.matrix};
    DeviceProgram program(device);program.limit_workspace(64*1024*1024);
    const auto local_before=program.retained_tensor_bytes(),owner_before=program.retained_tensor_bytes(held);
    WeightedRowGraph graph(program,owners,banks,error,available,held);
    graph.projection(graph.input(input),parameters,"witness.matrix");
    const auto local_delta=program.retained_tensor_bytes()-local_before;
    const auto owner_delta=program.retained_tensor_bytes(held)-owner_before;
    if(owner_delta<0||owner_delta>=available||local_delta<=available)
      throw std::runtime_error("borrowed storage witness did not isolate the local/global accounting boundary");
    const auto borrowed_first_use=local_delta-owner_delta;
    if(borrowed_first_use<parameters.matrix.storage().nbytes())
      throw std::runtime_error("borrowed parameter bank missing from accounting difference");
    program.close();
    DeviceProgram tight(device);tight.limit_workspace(64*1024*1024);bool rejected=false;std::string refusal;
    try{WeightedRowGraph bounded(tight,owners,banks,error,owner_delta-1,held);
      bounded.projection(bounded.input(input),parameters,"witness.matrix");}
    catch(const std::invalid_argument& e){refusal=e.what();rejected=refusal.find("local row graph tensor budget exceeded")!=std::string::npos
      &&refusal.find("owner_storage_increment=")!=std::string::npos;}
    if(!rejected)throw std::runtime_error("new storage did not enforce exact local allowance: "+refusal);
    tight.close();
    // A missing borrow declaration is conservative, never an uncharged bank.
    DeviceProgram undeclared(device);undeclared.limit_workspace(64*1024*1024);rejected=false;
    try{WeightedRowGraph bounded(undeclared,owners,banks,error,available,{input});
      bounded.projection(bounded.input(input),parameters,"undeclared.matrix");}
    catch(const std::invalid_argument& e){rejected=std::string(e.what()).find("local row graph tensor budget exceeded")!=std::string::npos;}
    if(!rejected)throw std::runtime_error("undeclared parameter bank escaped local accounting");undeclared.close();
    std::cout<<"weighted-budget: borrowed_owner_charged_once=true real_shortfall_refused=true undeclared_borrow_charged=true allowance="<<available<<" caller_bank_bytes="<<parameters.matrix.storage().nbytes()
      <<" local_program_delta="<<local_delta<<" genuinely_new_owner_storage="<<owner_delta
      <<" borrowed_first_use_bytes="<<borrowed_first_use<<" message="<<refusal<<'\n';return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
