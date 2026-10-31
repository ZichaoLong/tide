#pragma once
#include "benchmark.h"
#include "tide/checkpoint.h"
namespace weighted_benchmark {
struct Request {
  std::filesystem::path input,output,limits,resume;
  std::string device,implementation="native",mode="training",schedule="prefill",precision="float32",
    placement="native",scoring,packing="auto",return_packing="scalar",validation="strict",
    return_mode="complete",communication="ordinary";
  Index steps=3,warmup=1,threads=1,interop=1,workers=1;
  int64_t checkpoint_bytes=kDefaultCheckpointMaxBytes;
  bool checkpoint=false,observe_work=false,check_roots=false,help=false;
  std::vector<std::string> peers,nodes;
  double loss_scale=.001,lr=.001,momentum=.3,weight_decay=.01;
  at::ScalarType dtype() const {return precision=="float64"?at::kDouble:precision=="float16"?at::kHalf:at::kFloat;}
  Json task() const;
};
Request parse_request(int,char**);
void help();
}  // namespace weighted_benchmark
