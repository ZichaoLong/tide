#pragma once
#include "reverse_budget.h"
#include <map>
#include <string>
namespace tide::device_online {
using ReverseStatistics=std::map<std::string,int64_t>;
inline void merge_reverse_statistics(ReverseStatistics& into,const ReverseStatistics& from) {
  for(const auto& [key,value]:from) {
    auto [it,new_key]=into.emplace(key,value);if(new_key)continue;
    const auto suffix=key.substr(key.size()-4);
    if(suffix=="_min")it->second=std::min(it->second,value);
    else if(suffix=="_max")it->second=std::max(it->second,value);
    else it->second+=value;
  }
}
inline void record_reverse_plan(ReverseStatistics& into,const ReverseBatchPlan& p,const std::string& kind) {
  const auto prefix="reverse_"+kind+"_";
  ReverseStatistics stats{{prefix+"groups",1},{prefix+"requested_max",p.requested_rows},
    {prefix+"owner_rows_min",p.owner_rows},{prefix+"owner_rows_max",p.owner_rows},
    {prefix+"query_rows_min",p.query_rows},{prefix+"query_rows_max",p.query_rows},
    {prefix+"key_rows_min",p.key_rows},{prefix+"key_rows_max",p.key_rows},
    {prefix+"tensor_bytes",p.tensor_bytes},{prefix+"budget_bytes",p.budget}};
  merge_reverse_statistics(into,stats);
}
} // namespace tide::device_online
