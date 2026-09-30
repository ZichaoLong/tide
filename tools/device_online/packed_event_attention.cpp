#include "packed_event_attention.h"
#include "cann_api.h"
#include "aclrtlaunch_tide_event_plan.h"
#include "aclrtlaunch_tide_event_payload.h"
#include "aclrtlaunch_tide_event_indices.h"
#include "aclrtlaunch_tide_event_cache.h"
#include "aclrtlaunch_tide_fiber_chunk.h"
#include <cmath>

namespace tide::device_online {
namespace {uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}}
EventGroupStage EventAttentionGroup::propose(CannProgram& p,const ReadyBatch& ready,const ContentBatch& content,
    const at::Tensor& values,const at::Tensor& error) {
  const auto w=width,kv=kv_width,h=query_heads,kh=kv_heads,d=head_width,cap=capacity,c=chunk,ps=parameters,n=rows,os=owners;
  const auto opts=live.key.options(),longs=live.lengths.options();const auto old=live;
  const auto map=mapping,window=windows,cfg=config;
  EventGroupStage out;out.events=at::zeros({n,7},longs);out.counts=at::zeros({2},longs);
  // Zero sentinels belong only to padding. Real proposed rows are produced by
  // retained-tail copies and actual QKV; no stale row is visible to attention.
  out.cache={at::zeros_like(live.key),at::zeros_like(live.value),{}}; // Proposed lengths live in events[:,4].
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_event_plan)(1,stream,
    ptr(ready.fibers),ptr(ready.counts),ptr(map),ptr(old.lengths),ptr(window),ptr(out.events),ptr(out.counts),ptr(error),ps,cap,n),"plan event attention/window");},
    {ready.fibers,ready.counts,map,old.lengths,window,out.events,out.counts,error});
  auto dummy=at::zeros({1},longs),inactive=at::zeros({n},opts.dtype(at::kBool));
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_event_cache)(32,stream,
    ptr(out.events),ptr(out.counts),ptr(cfg),ptr(inactive),ptr(ready.fibers),ptr(out.cache.key),ptr(out.cache.value),ptr(old.key),ptr(old.value),
    ptr(old.lengths),ptr(dummy),ptr(dummy),ptr(dummy),ptr(dummy),ptr(error),kv,cap,int64_t(0),int64_t(0)),"pack retained event KV");},
    {out.events,out.counts,cfg,inactive,ready.fibers,out.cache.key,out.cache.value,old.key,old.value,old.lengths,dummy,error});
  auto source=at::empty({c},longs),parameter=at::empty_like(source),destination=at::empty_like(source),ids=at::empty_like(source);
  auto cursor=at::zeros({1},longs),zero=at::zeros_like(cursor),branch=at::zeros_like(error),tokens=at::zeros({1,4},longs);
  auto contents=at::zeros({n+1,w},opts);p.copy(contents.narrow(0,0,n),content.content);
  auto x=at::empty({c,1,w},opts),weight=at::empty({c,w,w+2*kv},opts),projected=at::empty({c,1,w+2*kv},opts),queries=at::empty({c,w},opts);
  auto indices=at::empty({c,h,cap},longs),additive=at::empty({c,1,1,cap},opts);
  auto keys=at::empty({c,h,cap,d},opts),v=at::empty_like(keys),kt=at::empty({c,h,d,cap},opts);
  auto scores=at::empty({c,h,1,cap},opts),logits=at::empty_like(scores),prob=at::empty_like(scores),result=at::empty({c,h,1,d},opts);
  auto out_weight=at::empty({c,w,w},opts),output=at::empty({c,1,w},opts),scale=at::full({},1./std::sqrt(double(d)),opts);
  const auto count_chunks=chunks;
  auto begin=p.label(),work=p.label(),end=p.label();p.copy(cursor,zero);p.mark(begin);
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_fiber_chunk)(1,stream,
    ptr(out.events),ptr(tokens),ptr(out.counts),ptr(dummy),ptr(cursor),ptr(source),ptr(parameter),ptr(destination),ptr(ids),ptr(branch),ptr(count_chunks),ptr(error),
    n,ps,c,int64_t(2),int64_t(0)),"pack actual event queries");},
    {out.events,tokens,out.counts,dummy,cursor,source,parameter,destination,ids,branch,count_chunks,error});
  p.branch(branch,{end,work});p.mark(work);
  p.index_select(contents,0,source,x.reshape({c,w}));p.index_select(qkv,0,parameter,weight);p.batch_matmul(x,weight,projected);
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_event_payload)(32,stream,
    ptr(out.events),ptr(ids),ptr(projected),ptr(queries),ptr(out.cache.key),ptr(out.cache.value),ptr(error),w,kv,cap,c),"place compact event QKV");},
    {out.events,ids,projected,queries,out.cache.key,out.cache.value,error});
  // Scalar metadata stores share cache lines even when their words differ.
  // One writer avoids cross-core cache-line writeback races in indices/masks.
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_event_indices)(1,stream,
    ptr(out.events),ptr(ids),ptr(indices),ptr(additive),ptr(error),h,kh,cap,os,c),"gather GQA head visibility");},
    {out.events,ids,indices,additive,error});
  p.index_select(out.cache.key.reshape({-1,d}),0,indices.reshape({-1}),keys.reshape({-1,d}));
  p.index_select(out.cache.value.reshape({-1,d}),0,indices.reshape({-1}),v.reshape({-1,d}));
  p.permute(keys,{0,1,3,2},kt);
  p.batch_matmul(queries.reshape({c*h,1,d}),kt.reshape({c*h,d,cap}),scores.reshape({c*h,1,cap}));
  p.multiply(scores,scale,logits);p.add(logits,additive);p.softmax(logits,3,prob);
  p.batch_matmul(prob.reshape({c*h,1,cap}),v.reshape({c*h,cap,d}),result.reshape({c*h,1,d}));
  p.index_select(projection,0,parameter,out_weight);p.batch_matmul(result.reshape({c,1,w}),out_weight,output);
  p.index_copy(values,0,destination,output.reshape({c,w}));p.branch(branch,{begin});p.mark(end);
  if(journal) {
    auto meta=at::empty_like(journal->meta),data=at::empty_like(journal->values),size=at::zeros({1},longs);const auto trace_rows=meta.size(0);
    p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_event_cache)(1,stream,
      ptr(out.events),ptr(out.counts),ptr(cfg),ptr(inactive),ptr(ready.fibers),ptr(out.cache.key),ptr(out.cache.value),ptr(old.key),ptr(old.value),
      ptr(old.lengths),ptr(dummy),ptr(meta),ptr(data),ptr(size),ptr(error),kv,cap,trace_rows,int64_t(2)),"record event KV observables");},
      {out.events,out.counts,cfg,inactive,ready.fibers,out.cache.key,out.cache.value,old.key,old.value,old.lengths,dummy,meta,data,size,error});
    out.journal=journal->propose(p,meta,data,size,error);
  }
  return out;
}
void EventAttentionGroup::commit(CannProgram& p,const EventGroupStage& out,const SelectionProposal& selection,const at::Tensor& error) {
  const auto old=live;const auto cfg=config,maximum=peak;const auto kv=kv_width,cap=capacity;auto dummy=at::zeros({1},live.lengths.options());
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_event_cache)(32,stream,
    ptr(out.events),ptr(out.counts),ptr(cfg),ptr(selection.active),ptr(dummy),ptr(out.cache.key),ptr(out.cache.value),ptr(old.key),ptr(old.value),
    ptr(old.lengths),ptr(maximum),ptr(dummy),ptr(dummy),ptr(dummy),ptr(error),kv,cap,int64_t(0),int64_t(1)),"commit selected event KV");},
    {out.events,out.counts,cfg,selection.active,out.cache.key,out.cache.value,old.key,old.value,old.lengths,maximum,dummy,error});
  if(journal)journal->commit(p,out.journal,error);
}
EventAttentionStage PackedEventAttention::propose(CannProgram& p,const ReadyBatch& ready,const ContentBatch& content,const at::Tensor& error,
    const at::Tensor& initial_values) {
  EventAttentionStage out;out.values=at::zeros({rows_+chunk_,width_},content.content.options());
  // Preserve actual proposals from other attention adapters. Each group then
  // overwrites only its own ready rows; adding scratch arrays would mix stale
  // inactive rows after the ready layout changes on a later device iteration.
  if(initial_values.defined())p.copy(out.values.narrow(0,0,rows_),initial_values);
  for(auto& g:groups_)out.groups.push_back(g->propose(p,ready,content,out.values,error));
  out.values=out.values.narrow(0,0,rows_);return out;
}
void PackedEventAttention::commit(CannProgram& p,const EventAttentionStage& out,const SelectionProposal& selection,const at::Tensor& error) {
  for(size_t i=0;i<groups_.size();++i)groups_[i]->commit(p,out.groups[i],selection,error);
}
} // namespace tide::device_online
