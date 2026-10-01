#include "packed_fiber_attention.h"
#include "tiled_attention.h"
#include "cann_api.h"
#include "aclrtlaunch_tide_fiber_plan.h"
#include "aclrtlaunch_tide_fiber_chunk.h"
#include "aclrtlaunch_tide_fiber_payload.h"
#include "aclrtlaunch_tide_fiber_indices.h"
#include "aclrtlaunch_tide_fiber_trace.h"
#include "aclrtlaunch_tide_fiber_commit.h"
#include <cmath>

namespace tide::device_online {
namespace {uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}}
FiberStage PackedFiberAttention::propose(CannProgram& p,const ContentProfile& profile,const ReadyBatch& ready,
    const ContentBatch& content,const ContentState& state,const at::Tensor& error) {
  const auto width=width_,rows=rows_,capacity=capacity_,chunk=chunk_,parameters=parameters_,owners=owners_,nodes=nodes_,ticks=max_ticks_;
  const auto opts=cache_.key.options(),longs=cache_.lengths.options();const int64_t fp16=cache_.key.scalar_type()==at::kHalf;
  FiberStage out;
  out.values=at::zeros({rows+chunk,width},opts);
  out.events=at::zeros({rows,7},longs);out.tokens=at::zeros({rows,4},longs);out.counts=at::zeros({2},longs);
  out.query_bias=at::empty({rows,capacity},opts);
  out.cache={at::empty_like(cache_.key),at::empty_like(cache_.value),at::empty_like(cache_.bias),at::empty_like(cache_.lengths)};
  p.copy(out.cache.key,cache_.key);p.copy(out.cache.value,cache_.value);p.copy(out.cache.bias,cache_.bias);
  p.copy(out.cache.lengths,cache_.lengths);
  const auto mapping=mapping_,policy=profile.clock_policy,sources=profile.sources,config=config_;
  const auto live=cache_;const auto inputs=int64_t(profile.graph.inputs.size());
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_fiber_plan)(1,stream,
    ptr(ready.fibers),ptr(ready.fiber_offsets),ptr(ready.counts),ptr(ready.atoms.coordinates),ptr(sources),ptr(mapping),
    ptr(live.lengths),ptr(state.clocks),ptr(policy),ptr(config),ptr(out.events),ptr(out.tokens),ptr(out.counts),ptr(error),
    nodes,parameters,capacity,rows,inputs,ticks),"plan complete fiber attention");},
    {ready.fibers,ready.fiber_offsets,ready.counts,ready.atoms.coordinates,sources,mapping,live.lengths,state.clocks,policy,config,
     out.events,out.tokens,out.counts,error});
  auto source=at::empty({chunk},longs),parameter=at::empty_like(source),destination=at::empty_like(source),ids=at::empty_like(source);
  auto cursor=at::zeros({1},longs),zero=at::zeros_like(cursor),branch=at::zeros_like(error);
  auto queries=at::zeros({rows+1,width},opts),query_output=at::zeros({rows+chunk,width},opts),pooled=at::zeros({rows+1,width},opts);
  const auto heads=heads_,decay=decay_,chunks=chunks_;
  std::vector<float> factors(parameters);
  for(int64_t n=0;n<nodes;++n)if(node_map_[n]>=0)
    factors[node_map_[n]]=float(1.0/std::sqrt(double(width/node_heads_[n])));
  auto scales=at::tensor(factors,at::kFloat).to(opts.device());
  // Sum-only graphs retain their original path. Other profiles share the same
  // QKV/cache work and apply coefficients only to the completed query outputs.
  const auto pool_kinds=pool_?pool_->kinds():at::zeros({parameters},longs);
  auto coefficients=at::ones({rows},opts.dtype(at::kFloat));
  auto payload=[&](int64_t mode,const at::Tensor& projected,const at::Tensor& q) {
    p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_fiber_payload)(32,stream,
      ptr(out.events),ptr(out.tokens),ptr(out.counts),ptr(ids),ptr(heads),ptr(scales),ptr(projected),ptr(q),
      ptr(out.cache.key),ptr(out.cache.value),ptr(out.cache.bias),ptr(out.query_bias),ptr(decay),ptr(pool_kinds),ptr(coefficients),ptr(pooled),ptr(error),width,capacity,chunk,mode,fp16),"packed fiber payload");},
      {out.events,out.tokens,out.counts,ids,heads,scales,projected,q,out.cache.key,out.cache.value,out.cache.bias,out.query_bias,decay,pool_kinds,coefficients,pooled,error});
  };
  payload(0,queries,queries);
  auto plan=[&](int64_t mode,int64_t target) {
    p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_fiber_chunk)(1,stream,
      ptr(out.events),ptr(out.tokens),ptr(out.counts),ptr(heads),ptr(cursor),ptr(source),ptr(parameter),ptr(destination),
      ptr(ids),ptr(branch),ptr(chunks),ptr(error),rows,parameters,chunk,mode,target),"pack fiber work chunk");},
      {out.events,out.tokens,out.counts,heads,cursor,source,parameter,destination,ids,branch,chunks,error});
  };
  auto weighted=at::zeros({rows+1,width},opts);p.copy(weighted.narrow(0,0,rows),content.weighted);
  auto x=at::empty({chunk,1,width},opts),weight=at::empty({chunk,width,3*width},opts);
  auto qkv=at::empty({chunk,1,3*width},opts),bias=at::empty({chunk,1,3*width},opts);
  auto head=p.label(),body=p.label(),done=p.label();p.copy(cursor,zero);p.mark(head);plan(0,0);p.branch(branch,{done,body});p.mark(body);
  p.index_select(weighted,0,source,x.reshape({chunk,width}));p.index_select(qkv_,0,parameter,weight);
  p.index_select(qkv_bias_,0,parameter,bias.reshape({chunk,3*width}));p.batch_matmul(x,weight,qkv);p.add(qkv,bias);
  payload(1,qkv,queries);p.branch(branch,{head});p.mark(done);
  auto bias32=fp16?at::empty(out.query_bias.sizes(),opts.dtype(at::kFloat)):out.query_bias;
  if(fp16)p.cast(out.query_bias,bias32);
  for(const auto h:head_groups_) {
    const auto d=width/h;
    auto q=at::empty({chunk,width},opts);
    auto begin=p.label(),work=p.label(),end=p.label();p.copy(cursor,zero);p.mark(begin);plan(1,h);p.branch(branch,{end,work});p.mark(work);
    p.index_select(queries,0,source,q);
    at::Tensor result;
    if(key_rows_<capacity) {
      result=append_tiled_attention(p,out.events,out.tokens,ids,q,out.cache.key,out.cache.value,
        bias32,error,key_work_,{h,h,capacity,owners,key_rows_,true,1.,0,true});
    }else {
      auto indices=at::empty({chunk,capacity},longs),additive=at::empty({chunk,1,1,capacity},opts.dtype(at::kFloat));
      auto keys=at::empty({chunk,capacity,width},opts),values=at::empty_like(keys);
      auto kt=at::empty({chunk,h,d,capacity},opts),vt=at::empty({chunk,h,capacity,d},opts);
      p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_fiber_indices)(1,stream,
        ptr(out.events),ptr(out.tokens),ptr(ids),ptr(bias32),ptr(indices),ptr(additive),ptr(error),capacity,owners,chunk),"gather complete KV visibility");},
        {out.events,out.tokens,ids,bias32,indices,additive,error});
      p.index_select(out.cache.key,0,indices.reshape({-1}),keys.reshape({chunk*capacity,width}));
      p.index_select(out.cache.value,0,indices.reshape({-1}),values.reshape({chunk*capacity,width}));
      p.permute(keys.reshape({chunk,capacity,h,d}),{0,2,3,1},kt);
      p.permute(values.reshape({chunk,capacity,h,d}),{0,2,1,3},vt);
      result=append_dense_attention(p,q.reshape({chunk,h,d}),kt,vt,additive,1.);
    }
    p.index_copy(query_output,0,destination,result.reshape({chunk,width}));p.branch(branch,{begin});p.mark(end);
  }
  if(pool_)coefficients=pool_->append(p,out.events,out.tokens,out.counts,ready,error,chunks);
  payload(2,queries,query_output);
  auto ow=at::empty({chunk,width,width},opts),ob=at::empty({chunk,1,width},opts),value=at::empty_like(x);
  auto begin=p.label(),work=p.label(),end=p.label();p.copy(cursor,zero);p.mark(begin);plan(2,0);p.branch(branch,{end,work});p.mark(work);
  p.index_select(pooled,0,source,x.reshape({chunk,width}));p.index_select(projection_,0,parameter,ow);
  p.index_select(projection_bias_,0,parameter,ob.reshape({chunk,width}));p.batch_matmul(x,ow,value);p.add(value,ob);
  p.index_copy(out.values,0,destination,value.reshape({chunk,width}));p.branch(branch,{begin});p.mark(end);
  if(journal_) {
    auto meta=at::empty_like(journal_->meta),values=at::empty_like(journal_->values),count=at::zeros({1},longs);
    const auto trace_rows=meta.size(0);
    p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_fiber_trace)(1,stream,
      ptr(out.events),ptr(out.counts),ptr(ready.fibers),ptr(live.key),ptr(live.value),ptr(live.bias),ptr(out.cache.key),ptr(out.cache.value),ptr(out.query_bias),
      ptr(meta),ptr(values),ptr(count),ptr(error),width,capacity,trace_rows,fp16),"record fiber cache observables");},
      {out.events,out.counts,ready.fibers,live.key,live.value,live.bias,out.cache.key,out.cache.value,out.query_bias,meta,values,count,error});
    out.journal=journal_->propose(p,meta,values,count,error);
  }
  out.values=out.values.narrow(0,0,rows);return out;
}
void PackedFiberAttention::commit(CannProgram& p,const FiberStage& out,const SelectionProposal& selection,const at::Tensor& error) {
  const int64_t fp16=cache_.key.scalar_type()==at::kHalf;
  const auto config=config_,peak=peak_;const auto live=cache_;const auto width=width_,capacity=capacity_;
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_fiber_commit)(32,stream,
    ptr(out.events),ptr(out.counts),ptr(config),ptr(selection.active),ptr(out.cache.key),ptr(out.cache.value),ptr(out.cache.bias),
    ptr(live.key),ptr(live.value),ptr(live.bias),ptr(live.lengths),ptr(peak),ptr(error),width,capacity,fp16),"commit selected fiber cache");},
    {out.events,out.counts,config,selection.active,out.cache.key,out.cache.value,out.cache.bias,live.key,live.value,live.bias,live.lengths,peak,error});
  if(journal_)journal_->commit(p,out.journal,error);
}
} // namespace tide::device_online
