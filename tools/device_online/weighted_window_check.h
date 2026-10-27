#pragma once
#include "weighted_window_vjp.h"
#include "tide/greedy.h"
namespace tide::device_online::window_check {
void run_window(at::Device,Graph,Model,std::vector<External>,const std::string& label,bool split,bool prefill);
using Records=std::map<std::string,at::Tensor>;
void seed(WeightedWindowAdjoint&,const Records&,const Result&,const std::string&);
void verify(const Graph&,const Model&,const Continuation&,const std::vector<External>&,
  const Result&,WeightedWindowAdjoint&,const Records&,const std::string&);
const at::Tensor& field(const State&,const std::string&);
int64_t stage(const Records&,int64_t node,int64_t sample,int64_t time);
}
