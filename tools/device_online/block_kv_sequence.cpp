#include "block_kv_sequence.h"
#include "weighted_storage_budget.h"
#include "device_launch_tide_block_kv_sequence.h"
#include <limits>

namespace tide::device_online {
namespace {
uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}
void check(const at::Tensor& x,const at::Tensor& e,at::ScalarType type,at::IntArrayRef shape){
  if(!x.defined()||x.device()!=e.device()||x.scalar_type()!=type||x.sizes()!=shape||
    !x.is_contiguous()||x.requires_grad())throw std::invalid_argument("invalid KV sequence tensor");
}
void admit(long double bytes,int64_t budget){if(budget<1||bytes>budget)throw std::invalid_argument("KV sequence tensor budget exceeded");}
}
BlockKVSequenceTape append_block_kv_sequence(DeviceProgram& p,const weighted::KVConfig& config,
    const WeightedRawState& initial,const at::Tensor& content,const at::Tensor& times,
    const at::Tensor& owners,const at::Tensor& present,const WeightedKVParameters& weights,
    const at::Tensor& error,int64_t budget){
  config.validate();
  if(!present.defined()||present.dim()!=3||!initial.integer.defined()||initial.integer.dim()!=2||initial.integer.size(1)<=4)
    throw std::invalid_argument("KV sequence needs B,L,N coordinates and bounded raw initial state");
  const auto b=present.size(0),l=present.size(1),n=present.size(2);
  if(b<1||l<1||n<1||b>std::numeric_limits<int64_t>::max()/n||b*n>std::numeric_limits<int64_t>::max()/l)
    throw std::invalid_argument("KV sequence dimensions overflow");
  const int64_t rows=b*l*n,capacity=initial.integer.size(1)-4;
  WeightedRawLayout layout(config,capacity),one(config,1);validate_weighted_raw_buffers(layout,initial);
  check(error,error,at::kInt,{1});check(present,error,at::kBool,{b,l,n});
  check(times,error,at::kLong,{b,l,n});check(owners,error,at::kLong,{b,l,n});
  check(content,error,at::kFloat,{rows,config.width+1});
  check(initial.floating,error,at::kFloat,{b*n,layout.floating_width()});
  check(initial.integer,error,at::kLong,{b*n,layout.integer_width()});
  const auto fw=layout.floating_width(),iw=layout.integer_width(),ww=one.floating_width(),count=int64_t(layout.fields().size());
  const WeightedStorageIncrement storage(p,{initial.floating,initial.integer,content,times,owners,present,
    weights.base,weights.left,weights.right,weights.summary,error});
  const long double retained=rows*(8.L*fw+16.L*iw+16.L*capacity+4.L*ww+40)+b*n*(4.L*fw+16.L*iw)+32.L*count;
  admit(retained,budget);
  // Existing vendor projections produce every positive reception once. The
  // one-entry raw adapter is only a projection result, never persistent state.
  auto empty_values=at::zeros({rows,ww},content.options());
  auto empty_integer=at::zeros({rows,5},times.options());empty_integer.select(1,0).fill_(-1);
  auto write=append_weighted_kv_update(p,config,{empty_values,empty_integer},content,times.reshape({rows}),
    owners.reshape({rows}),present.reshape({rows}),weights,error,budget-static_cast<int64_t>(retained));
  BlockKVSequenceTape t{config,b,l,n,layout,std::move(write),{},{},{},{},{},{}};
  t.old={at::zeros({rows,fw},content.options()),at::zeros({rows,iw},times.options())};
  t.proposal={at::zeros_like(t.old.floating),at::zeros_like(t.old.integer)};
  t.final={at::zeros_like(initial.floating),at::zeros_like(initial.integer)};
  t.initial_integer=at::zeros_like(initial.integer);p.copy(t.initial_integer,initial.integer);
  std::vector<int64_t> fields;
  for(const auto& f:layout.fields())fields.insert(fields.end(),{f.offset,one.field(f.name).offset,f.width,int64_t(f.ragged)});
  t.fields=at::tensor(fields,at::kLong).to(error.device());t.mapping=at::full({2,rows,capacity},-1,times.options());
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_block_kv_sequence)(1,stream,
    ptr(t.fields),ptr(initial.floating),ptr(t.initial_integer),ptr(t.writes.raw.result.floating),ptr(t.writes.raw.result.integer),
    ptr(times),ptr(t.old.floating),ptr(t.old.integer),ptr(t.proposal.floating),ptr(t.proposal.integer),
    ptr(t.final.floating),ptr(t.final.integer),ptr(t.mapping),ptr(error),b,l,n,fw,ww,capacity,count,config.window),
    "whole-window exact KV reception sequence");},
    {t.fields,initial.floating,t.initial_integer,t.writes.raw.result.floating,t.writes.raw.result.integer,times,
      t.old.floating,t.old.integer,t.proposal.floating,t.proposal.integer,t.final.floating,t.final.integer,t.mapping,error});
  admit(storage.bytes(),budget);return t;
}
}
