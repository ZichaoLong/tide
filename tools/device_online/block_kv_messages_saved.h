#pragma once
#include "block_kv_messages.h"
#include "weighted_saved_tensors.h"

namespace tide::device_online::block_message_detail {
inline std::vector<at::Tensor> saved(const BlockKVMessageTape& t){
  auto out=saved_tensors(t.full);append_saved_tensors(out,t.selection.query);
  auto a=t.selection.describe.graph->saved_tensors(),b=t.columns->saved_tensors();
  out.insert(out.end(),a.begin(),a.end());out.insert(out.end(),b.begin(),b.end());
  const auto& s=t.selection;const auto& r=s.region;
  out.insert(out.end(),{s.owners,s.present,s.node_times,s.order,s.inverse,s.ids,s.times,s.lengths,s.node_region_row,s.controls,s.active,
    r.fields.received,r.fields.caps,r.fields.score,r.fields.evidence,r.fields.current,r.fields.key,
    r.lengths,r.ids,r.times,r.context_input,r.query_input,r.statistics,r.context_rows,r.query_rows,r.context_values,
    r.query_values,r.scores,r.bounds,r.parameters,r.controls,r.active,r.history_times,r.history_present,
    t.messages.strength,t.messages.content,t.messages.present,t.messages.times,t.messages.projection});
  for(const auto* h:{&r.context_head,&r.query_head})if(*h){const auto& x=**h;
    out.insert(out.end(),{x.input,x.owners,x.indices,x.preactivation,x.hidden,x.output,
      x.rows.matrix,x.rows.hidden_bias,x.rows.projection,x.rows.bias});}
  if(r.allocation){const auto& x=*r.allocation;out.insert(out.end(),{x.controls,x.slopes,x.statistics,x.branches,x.lengths,x.row_errors});}
  if(r.sending){const auto& x=*r.sending;out.insert(out.end(),{x.controls,x.details,x.statistics,x.branches,x.lengths});}
  return out;
}
}
