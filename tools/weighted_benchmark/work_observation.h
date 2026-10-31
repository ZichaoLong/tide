#pragma once
#include "benchmark.h"
#include "tide/operator_work.h"
namespace weighted_benchmark {
inline Json packing_work(){
  Json metrics=Json::object();
  for(const auto& [key,value]:tide::work::metrics())if(key.rfind("packing/",0)==0)metrics[key]=value;
  return {{"schema","tide-weighted-packing-observation-v1"},
    {"scope","eager local forward calls; training includes private forward replay; host metadata only, no additional tensor reads or device synchronization"},
    {"throughput_sample",false},{"metrics",metrics}};
}
}
