#include "block_kv_local.h"
#include "block_read_selection_internal.h"
#include "block_kv_local_saved.h"
#include "device_launch_tide_block_raw_roots.h"
namespace tide::device_online {
BlockKVLocalGradient append_block_kv_local_vjp(DeviceProgram& p,const BlockKVLocalTape& t,
    const BlockKVLocalRoots& roots,const at::Tensor& error,int64_t budget){
  using namespace block_selection_detail;
  const auto b=t.sequence.batch,l=t.sequence.length,n=t.sequence.nodes,rows=b*l*n,fw=t.sequence.layout.floating_width();
  const auto count=int64_t(t.sequence.layout.fields().size()),capacity=t.sequence.layout.capacity(),d=t.sequence.config.width;
  auto borrowed=block_local_detail::saved(t);borrowed.push_back(error);
  for(const auto* r:{&roots.messages.controls,&roots.messages.full,&roots.messages.message_strength,&roots.messages.message_content,&roots.messages.projection,&roots.received}){
    borrowed.push_back(r->value);borrowed.push_back(r->connected);}
  for(const auto* r:{&roots.messages.read.memory,&roots.messages.read.total}){borrowed.push_back(r->value);borrowed.push_back(r->connected);}
  for(const auto& [name,r]:roots.messages.read.describe){borrowed.push_back(r.value);borrowed.push_back(r.connected);}
  for(const auto& x:{roots.raw.old,roots.raw.old_connected,roots.raw.proposal,roots.raw.proposal_connected,roots.raw.final,roots.raw.final_connected})borrowed.push_back(x);
  const WeightedStorageIncrement storage(p,std::move(borrowed));auto remaining=[&](){admit(storage.bytes(),budget);return budget-storage.bytes();};
  admit(rows*(8.L*fw+2.L*count)+b*n*(4.L*fw+count),remaining());
  auto message=append_block_kv_messages_vjp(p,t.messages,roots.messages,error,remaining());
  BlockKVSequenceRoots combined{at::zeros_like(t.sequence.old.floating),at::zeros({rows,count},t.present.options()),
    at::zeros_like(t.sequence.proposal.floating),at::zeros({rows,count},t.present.options()),roots.raw.final,roots.raw.final_connected};
  p.zero(combined.old);p.zero(combined.old_connected);p.zero(combined.proposal);p.zero(combined.proposal_connected);
  if(!combined.final.defined()&&!combined.final_connected.defined()){
    combined.final=at::zeros_like(t.sequence.final.floating);combined.final_connected=at::zeros({b*n,count},t.present.options());}
  auto bound=at::zeros_like(t.sequence.final.floating),bound_flags=at::zeros({b*n,count},t.present.options());p.zero(bound);p.zero(bound_flags);
  const auto identity=at::arange(rows,t.sequence.mapping.options()),fields=t.sequence.fields;
  auto add=[&](const Tensor& values,const Tensor& connected,const Tensor& metadata,const Tensor& mapping,
      const Tensor& target,const Tensor& target_roots){
    if(values.defined()!=connected.defined())throw std::invalid_argument("local raw root requires cotangent and mask");
    if(!values.defined())return;
    check(values,error,at::kFloat,{rows,fw});check(connected,error,at::kBool,{rows,count});const auto targets=target.size(0);
    p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_block_raw_roots)(1,stream,
      ptr(fields),ptr(metadata),ptr(mapping),ptr(values),ptr(connected),ptr(target),ptr(target_roots),ptr(error),
      rows,targets,fw,capacity,count),"independent raw field contribution merge");},
      {fields,metadata,mapping,values,connected,target,target_roots,error});
  };
  add(roots.raw.old,roots.raw.old_connected,t.sequence.old.integer,identity,combined.old,combined.old_connected);
  add(roots.raw.proposal,roots.raw.proposal_connected,t.sequence.proposal.integer,identity,combined.proposal,combined.proposal_connected);
  add(message.full.raw,message.full.raw_connected,t.sequence.proposal.integer,identity,combined.proposal,combined.proposal_connected);
  if(t.read_mode=="+")add(message.read.raw,message.read.raw_connected,t.read.integer,identity,combined.proposal,combined.proposal_connected);
  else if(t.read_mode=="-")add(message.read.raw,message.read.raw_connected,t.read.integer,identity,combined.old,combined.old_connected);
  else add(message.read.raw,message.read.raw_connected,t.read.integer,t.initial_mapping,bound,bound_flags);
  auto sequence=append_block_kv_sequence_vjp(p,t.sequence,combined,error,remaining());
  std::vector<WeightedRowRoot> incoming{
    {t.received_index,sequence.update.content,sequence.update.content_connected},
    {t.mass_index,message.read.mass.reshape({rows,1}),message.read.mass_connected.reshape({rows})},
    {t.content_index,message.content.value.reshape({rows,d}),message.content.connected.reshape({rows})}};
  if(roots.received.value.defined()!=roots.received.connected.defined())throw std::invalid_argument("received root requires cotangent and mask");
  if(roots.received.value.defined())incoming.push_back({t.received_index,roots.received.value,roots.received.connected});
  auto received=t.received->reverse(p,incoming,error,remaining()).rows.at(t.received_index);
  auto aggregate=append_weighted_aggregate_vjp(p,t.aggregate,received.value,received.connected,error,remaining());
  admit(storage.bytes(),budget);
  return {std::move(aggregate),std::move(sequence),std::move(message),bound,bound_flags};
}
}
