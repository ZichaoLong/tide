#include "block_kv_local.h"
#include "block_read_selection_internal.h"
#include "block_kv_local_saved.h"
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
  if(!t.sequence.bank)throw std::invalid_argument("KV local reverse requires the actual shared bank tape");
  auto message=append_block_kv_messages_vjp(p,t.messages,roots.messages,error,remaining());
  if(!message.full_bank||!message.read.bank)throw std::logic_error("bank local lost Read/Full adjoints");
  std::vector<BlockKVBankAdjoint> contributions{*message.full_bank};
  auto bound=at::zeros_like(t.sequence.final.floating),bound_flags=at::zeros({b*n,count},t.present.options());
  if(t.read_mode=="0"){
    auto g=append_block_kv_bank_vjp(p,message.read.bank->snapshot.bank,{*message.read.bank},{},error,remaining());
    bound=g.initial;bound_flags=g.initial_connected;
  }else contributions.push_back(*message.read.bank);
  std::vector<BlockKVBankRawRoot> raw{
    {block_kv_snapshot(t.sequence,false),roots.raw.old,roots.raw.old_connected},
    {block_kv_snapshot(t.sequence,true),roots.raw.proposal,roots.raw.proposal_connected},
    {block_kv_final_snapshot(t.sequence),roots.raw.final,roots.raw.final_connected}};
  auto bank=append_block_kv_bank_vjp(p,t.sequence.bank,contributions,raw,error,remaining());
  auto update=append_weighted_kv_update_vjp(p,t.sequence.writes,bank.writes,bank.write_connected,error,remaining());
  BlockKVSequenceGradient sequence{bank.initial,bank.initial_connected,std::move(update)};
  std::vector<WeightedRowRoot> incoming{
    {t.received_index,sequence.update.content,sequence.update.content_connected},
    {t.mass_index,message.read.mass.reshape({rows,1}),message.read.mass_connected.reshape({rows})},
    {t.content_index,message.content.value.reshape({rows,d}),message.content.connected.reshape({rows})}};
  if(roots.received.value.defined()!=roots.received.connected.defined())throw std::invalid_argument("received root requires cotangent and mask");
  if(roots.received.value.defined())incoming.push_back({t.received_index,roots.received.value,roots.received.connected});
  auto received=t.received->reverse(p,incoming,error,remaining()).rows.at(t.received_index);
  auto aggregate=append_weighted_aggregate_vjp(p,t.aggregate,received.value,received.connected,error,remaining());
  std::optional<WeightedSourceAdaptationGradient> adaptation;
  if(t.source_adaptation){if(!aggregate.parameters)throw std::logic_error("active source adaptation lost Aggregate adjoints");
    adaptation=append_weighted_source_adaptation_vjp(p,*t.source_adaptation,*aggregate.parameters,aggregate.owner_connected,error,remaining());}
  admit(storage.bytes(),budget);
  return {std::move(aggregate),std::move(sequence),std::move(message),bound,bound_flags,std::move(adaptation)};
}
}
