#pragma once
#include "block_kv_local.h"
#include "block_kv_messages_saved.h"
namespace tide::device_online::block_local_detail {
inline std::vector<at::Tensor> saved(const BlockKVSequenceTape& t){
  const auto& w=t.writes;const auto& r=w.raw;
  std::vector<at::Tensor> out{t.old.floating,t.old.integer,t.proposal.floating,t.proposal.integer,t.final.floating,t.final.integer,
    t.fields,t.mapping,t.initial_integer,r.result.floating,r.result.integer,r.old_integer,r.proposal,r.fields,r.mapping,r.receive,r.written,
    w.saved_rows.base,w.saved_rows.left,w.saved_rows.right,w.saved_rows.summary,w.owners,w.kv_indices,w.summary_indices};
  for(const auto* p:{&w.kv,&w.summary,&w.adapter_low,&w.adapter_up})append_saved_tensors(out,*p);return out;
}
inline std::vector<at::Tensor> saved(const BlockKVLocalTape& t){
  auto out=block_message_detail::saved(t.messages);auto sequence=saved(t.sequence);out.insert(out.end(),sequence.begin(),sequence.end());
  auto rows=t.received->saved_tensors();out.insert(out.end(),rows.begin(),rows.end());
  const auto& a=t.aggregate;
  for(const auto& x:{a.payload,a.lengths,a.owners,a.clean,a.normal,a.rstd,a.total,a.values,a.output,t.read.floating,t.read.integer,t.initial_mapping,t.present})out.push_back(x);
  if(a.source){const auto& h=*a.source;for(const auto& x:{h.input,h.owners,h.indices,h.preactivation,h.hidden,h.output,h.rows.matrix,h.rows.hidden_bias,h.rows.projection,h.rows.bias})out.push_back(x);}
  if(t.source_adaptation){const auto& s=*t.source_adaptation;out.insert(out.end(),{s.effective.a,s.effective.b,s.effective.embedding,s.saved.a_left,s.saved.a_right,s.saved.b_left,s.saved.b_right});}
  return out;
}
}
