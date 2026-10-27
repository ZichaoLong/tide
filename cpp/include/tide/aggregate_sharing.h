#pragma once
#include "tide/aggregate.h"
namespace tide {
struct SharedAggregates {
  std::map<Index,std::vector<size_t>> compute;
  std::vector<std::pair<size_t,size_t>> copies;
};
SharedAggregates plan_aggregate_sharing(const Graph&,const Model&,const std::vector<Event>&,
    const std::map<Index,std::vector<size_t>>&,bool,std::map<std::string,Index>&);
void decode_shared_aggregates(const Graph&,const Model&,std::vector<Event>&,const SharedAggregates&);
}
