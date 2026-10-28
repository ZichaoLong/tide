// Adapter only: serialize the actual failing native row without any re-solve.
#include "benchmark.h"
#include "tide/allocation_failure.h"
#include <cmath>
namespace weighted_benchmark {
Json allocation_failure(const tide::weighted::AllocationFailure& error){
  Json fields=Json::object();
  for(const auto& [name,value]:error.tensors){
    auto cpu=value.to(at::kCPU,at::kDouble).contiguous().flatten();Json numbers=Json::array();
    for(Index i=0;i<cpu.numel();++i){auto v=cpu[i].item<double>();
      if(std::isfinite(v))numbers.push_back(v);
      else numbers.push_back(std::isnan(v)?"nan":v>0?"inf":"-inf");
    }
    fields[name]={{"dtype",value.scalar_type()==at::kDouble?"float64":"float32"},
      {"device",value.device().str()},{"shape",value.sizes().vec()},{"values",numbers}};
  }
  Json record{{"schema","tide-allocation-failure-v1"},{"error",error.what()},{"row",error.row},
    {"tensors",fields},{"details",error.details},{"nodes",error.nodes},{"members",error.members},{"time",nullptr}};
  if(error.time)record["time"]=*error.time;return record;
}
}
