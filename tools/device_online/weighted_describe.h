#pragma once
#include "weighted_row_graph.h"
#include "tide/weighted.h"
#include <memory>
namespace tide::device_online {
using WeightedDescribeParameters=std::map<std::string,at::Tensor>;
struct WeightedDescribeInput {
  at::Tensor mass,content,short_memory,long_memory,short_total,long_total;
};
struct WeightedDescribeTape {
  std::shared_ptr<WeightedRowGraph> graph;
  std::map<std::string,size_t> inputs,outputs;
  std::map<std::string,at::Tensor> fields;
};
struct WeightedDescribeGradient {
  std::map<std::string,WeightedRowAdjoint> inputs,parameters;
};
// Allocation §3/5.1, all formal scoring/evidence/key/gain configurations.
// Each shape includes its parameter-bank axis, with explicit scalar packing.
std::map<std::string,std::vector<int64_t>> weighted_describe_parameter_shapes(
  int64_t banks,int64_t width,int64_t summary_width,const weighted::RegionalConfig&);
WeightedDescribeTape append_weighted_describe(DeviceProgram&,const WeightedDescribeInput&,
  const at::Tensor& owners,const WeightedDescribeParameters&,const weighted::RegionalConfig&,
  double epsilon,const at::Tensor& error,int64_t budget=256*1024*1024);
WeightedDescribeGradient append_weighted_describe_vjp(DeviceProgram&,const WeightedDescribeTape&,
  const std::map<std::string,WeightedRowAdjoint>& roots,const at::Tensor& error,int64_t budget=256*1024*1024);
}
