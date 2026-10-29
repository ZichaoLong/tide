#pragma once
// Allocation §6/S.2: bounded opt-in snapshots of actual native calls. No Python
// dependency, reference recomputation, file I/O or implicit device-to-host copy.
#include "tide/weighted.h"
#include "tide/allocation_failure.h"

namespace tide::weighted {
using AllocationFields=std::map<std::string,Tensor>;
using AllocationDetails=std::map<std::string,std::string>;
struct AllocationObservations {
  bool enabled=false;
  Index max_rows=0,max_bytes=0,bytes=0,dropped=0;
  std::vector<std::pair<AllocationFields,AllocationDetails>> rows;
};
inline thread_local AllocationObservations allocation_observations;
inline void begin_allocation_observation(Index rows,Index bytes){
  if(rows<1||bytes<1||allocation_observations.enabled)
    throw std::invalid_argument("allocation observation requires positive bounds and no active observer");
  allocation_observations={true,rows,bytes,0,0,{}};
}
inline AllocationObservations end_allocation_observation(){
  if(!allocation_observations.enabled)throw std::invalid_argument("no active allocation observer");
  auto result=std::move(allocation_observations);allocation_observations={};return result;
}
inline AllocationFields allocation_fields(const AllocationRow& r,const Allocation& a){
  return {{"scores",r.scores},{"caps",r.caps},{"lower",r.lower},{"upper",r.upper},
          {"temperature",r.temperature},{"shift",r.shift},{"controls",a.controls},
          {"multiplier",a.multiplier},{"residual",a.residual},{"free",a.free}};
}
inline void observe_allocation(const AllocationRow& row,const Allocation& a,const std::string& kind,
                               std::optional<double> alpha,const std::string& solver){
  auto& out=allocation_observations;if(!out.enabled)return;
  auto fields=allocation_fields(row,a);long double bytes=0;
  for(const auto& [name,t]:fields)if(t.defined())bytes+=static_cast<long double>(t.numel())*t.element_size();
  if(out.rows.size()>=static_cast<size_t>(out.max_rows)||bytes>out.max_bytes-out.bytes){++out.dropped;return;}
  AllocationFields snapshot;
  for(const auto& [name,t]:fields)if(t.defined())snapshot[name]=t.detach().clone();
  out.bytes+=static_cast<Index>(bytes);
  out.rows.push_back({std::move(snapshot),{{"regularizer",kind},{"alpha",alpha?std::to_string(*alpha):"None"},
    {"solver",solver},{"branch",a.constraint},{"observation","successful actual return"}}});
}
inline AllocationFailure scalar_allocation_failure(const std::string& message,const AllocationRow& row,
    const Allocation& a,const Tensor& response,const std::string& kind,std::optional<double> alpha){
  auto fields=allocation_fields(row,a);if(response.defined())fields["response"]=response;
  return AllocationFailure(message,0,fields,{{"regularizer",kind},{"alpha",alpha?std::to_string(*alpha):"None"},
    {"branch",a.constraint},{"solver","native-scalar"},{"stage","numeric"}});
}
}
