#include "weighted_kv_block.h"
#include "weighted_query_vjp.h"
#include "tide/packing_work.h"
namespace tide::weighted::block {
namespace {
const State* state_of(const ReadInput& q){return q.state;}
const State* state_of(const FullInput& q){return q.comparison;}
template<class Input,class Make> std::map<size_t,TensorFields> dispatch(const NodeWeights& w,const std::vector<Input>& requests,
    const KVConfig& c,const RegionalConfig& r,bool full,Make make){
  std::vector<size_t> ids;std::vector<Row> rows;
  // A generic/short/positioned fallback must not copy every State map just
  // to discover that it has no block provenance. Inspect the borrowed handle.
  for(size_t i=0;i<requests.size();++i)if(view(state_of(requests[i]))){
    ids.push_back(i);rows.push_back(make(requests[i]));
  }
  auto p=plan(rows,c,full);if(!p)return {};
  auto values=evaluate(w,c,r,*p);std::map<size_t,TensorFields> result;
  for(size_t i=0;i<ids.size();++i)result[ids[i]]=std::move(values[i]);
  work::packing::projection(full?"kv_block_full":"kv_block_read",ids.size());return result;
}
}
std::vector<TensorFields> read(const NodeWeights& w,const std::vector<ReadInput>& requests,const KVConfig& c,
    const RegionalConfig& r,bool grad,bool heads){
  auto handled=dispatch(w,requests,c,r,false,[](const ReadInput& q){return Row{q.state?*q.state:State{},q.time,q.content.value,{}};});
  std::vector<ReadInput> rest;std::vector<size_t> ids;
  std::vector<TensorFields> result(requests.size());
  for(size_t i=0;i<requests.size();++i){if(handled.count(i))result[i]=std::move(handled.at(i));else{rest.push_back(requests[i]);ids.push_back(i);}}
  auto fallback=grad?query::read_vjp(w,rest,c,r,2,heads):query::describe(w,rest,c,r,2,heads);
  for(size_t i=0;i<ids.size();++i)result[ids[i]]=std::move(fallback[i]);return result;
}
std::vector<FullResult> full(const NodeWeights& w,const std::vector<FullInput>& requests,Index slots,const Options& o,
    const KVConfig& c,bool grad){
  detail::require(o.mode=="hard","weighted Full has exact first-order VJP");
  auto handled=dispatch(w,requests,c,RegionalConfig{},true,[](const FullInput& q){return Row{*q.comparison,q.time,q.content.value,q.control};});
  std::vector<FullInput> rest;std::vector<size_t> ids;std::vector<FullResult> result(requests.size());
  for(size_t i=0;i<requests.size();++i){if(handled.count(i)){auto& out=result[i];out.value=handled.at(i).at("value");
      for(Index j=0;j<slots;++j)out.emitted.push_back({j,out.value});}
    else{rest.push_back(requests[i]);ids.push_back(i);}}
  auto fallback=grad?query::full_vjp(w,rest,slots,o,c,2):query::full(w,rest,slots,o,c,2);
  for(size_t i=0;i<ids.size();++i)result[ids[i]]=std::move(fallback[i]);return result;
}
}
