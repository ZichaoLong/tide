#include "block_kv_sequence.h"
#include "weighted_storage_budget.h"
#include "device_launch_tide_block_kv_sequence_reverse.h"
namespace tide::device_online {
namespace {
uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}
void check(const at::Tensor& x,const at::Tensor& e,at::ScalarType type,at::IntArrayRef shape){
  if(!x.defined()||x.device()!=e.device()||x.scalar_type()!=type||x.sizes()!=shape||
    !x.is_contiguous()||x.requires_grad())throw std::invalid_argument("invalid KV sequence independent root");
}
void admit(long double bytes,int64_t budget){if(budget<1||bytes>budget)throw std::invalid_argument("KV sequence VJP tensor budget exceeded");}
}
BlockKVSequenceGradient append_block_kv_sequence_vjp(DeviceProgram& p,const BlockKVSequenceTape& t,
    const BlockKVSequenceRoots& roots,const at::Tensor& error,int64_t budget){
  const auto owners=t.batch*t.nodes,rows=owners*t.length,fw=t.layout.floating_width(),capacity=t.layout.capacity();
  const auto ww=t.writes.raw.layout.floating_width(),count=int64_t(t.layout.fields().size());
  if(owners<1||rows<1)throw std::invalid_argument("missing KV sequence tape");
  check(error,error,at::kInt,{1});
  check(roots.old,error,at::kFloat,{rows,fw});check(roots.proposal,error,at::kFloat,{rows,fw});
  check(roots.old_connected,error,at::kBool,{rows,count});check(roots.proposal_connected,error,at::kBool,{rows,count});
  check(roots.final,error,at::kFloat,{owners,fw});check(roots.final_connected,error,at::kBool,{owners,count});
  const WeightedStorageIncrement storage(p,{t.fields,t.mapping,t.initial_integer,t.old.integer,t.proposal.integer,
    t.writes.raw.result.floating,t.writes.raw.result.integer,roots.old,roots.old_connected,roots.proposal,
    roots.proposal_connected,roots.final,roots.final_connected,error});
  const long double retained=owners*(4.L*fw+count)+rows*(4.L*ww+count);admit(retained,budget);
  auto initial=at::zeros({owners,fw},roots.old.options()),initial_roots=at::zeros({owners,count},roots.old_connected.options());
  auto writes=at::zeros({rows,ww},roots.old.options()),write_roots=at::zeros({rows,count},roots.old_connected.options());
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_block_kv_sequence_reverse)(1,stream,
    ptr(t.fields),ptr(t.writes.raw.result.integer),ptr(t.old.integer),ptr(t.proposal.integer),ptr(t.mapping),
    ptr(roots.old),ptr(roots.old_connected),ptr(roots.proposal),ptr(roots.proposal_connected),ptr(roots.final),ptr(roots.final_connected),
    ptr(initial),ptr(initial_roots),ptr(writes),ptr(write_roots),ptr(error),t.batch,t.length,t.nodes,fw,ww,capacity,count),
    "KV prefix field transpose and independent structural roots");},
    {t.fields,t.writes.raw.result.integer,t.old.integer,t.proposal.integer,t.mapping,roots.old,roots.old_connected,
      roots.proposal,roots.proposal_connected,roots.final,roots.final_connected,initial,initial_roots,writes,write_roots,error});
  auto gradient=append_weighted_kv_update_vjp(p,t.writes,writes,write_roots,error,budget-static_cast<int64_t>(retained));
  admit(storage.bytes(),budget);return {initial,initial_roots,std::move(gradient)};
}
}
