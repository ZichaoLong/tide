#include "content_flow_internal.h"
#include <algorithm>
#include <array>
#include <stdexcept>

namespace tide::device_online {
namespace {
std::map<Owner,History> download_history(const Graph& g,const SelectionHistory& history,Index samples) {
  std::map<Owner,History> result;
  auto counts=history.counts.cpu(),seen=history.seen.cpu(),last=history.last_time.cpu(),has=history.present.cpu();
  for(Index b=0;b<samples;++b)for(Index r=0;r<Index(g.regions.size());++r)if(has[b][r].item<bool>()) {
    History h;h.last_time=last[b][r].item<Index>();h.node_maps["selected"]={};
    for(Index n=0;n<Index(g.nodes.size());++n)if(g.nodes[n].region==r&&seen[b][n].item<bool>())h.node_maps["selected"][n]=counts[b][n].item<Index>();
    result[{b,r}]=std::move(h);
  }
  return result;
}
}
void upload_atoms(const std::vector<Atom>& atoms,const AtomBatch& out) {
  if(atoms.size()>size_t(out.valid.numel()))throw std::invalid_argument("initial/input atoms exceed live capacity");
  std::vector<int64_t> coordinates;std::vector<Tensor> values;
  for(const auto& a:atoms){coordinates.insert(coordinates.end(),{a.batch,a.node,a.time,a.kind,a.source,a.position});values.push_back(a.value);}
  out.valid.zero_();
  if(atoms.empty())return;
  out.coordinates.narrow(0,0,atoms.size()).copy_(at::tensor(coordinates,at::kLong).reshape({-1,6}));
  out.values.narrow(0,0,atoms.size()).copy_(at::stack(values));out.valid.narrow(0,0,atoms.size()).fill_(true);
}
std::vector<Atom> download_atoms(const AtomBatch& q) {
  auto c=q.coordinates.cpu(),v=q.values.cpu(),live=q.valid.cpu();
  const auto coords=c.accessor<int64_t,2>();std::vector<Atom> out;
  for(int64_t i=0;i<live.numel();++i)if(live[i].item<bool>())
    out.push_back({coords[i][0],coords[i][1],coords[i][2],coords[i][3],coords[i][4],coords[i][5],v[i].clone()});
  std::stable_sort(out.begin(),out.end(),[](const Atom& a,const Atom& b){return a.key()<b.key();});return out;
}
Continuation ContentFlow::Impl::export_continuation() const {
  const auto& g=profile.graph;auto q=boundary;
  q.pending=download_atoms(pending->atoms());
  auto state_values=state.values.cpu(),clocks=state.clocks.cpu(),present=state.present.cpu();
  for(Index b=0;b<q.batch_size;++b)for(Index n=0;n<Index(g.nodes.size());++n)if(present[b][n].item<bool>())
    q.states[{b,n}]={state_values[b][n].clone(),clocks[b][n][0].item<Index>(),clocks[b][n][1].item<Index>()};
  if(attention)attention->export_states(q);
  if(event_attention)event_attention->export_states(q);
  q.history=download_history(g,history,q.batch_size);return q;
}
Result ContentFlow::Impl::export_result() const {
  // Explicit boundary-only materialization. This copy never feeds the device
  // continuation, including when a client skips exports for several windows.
  const auto& g=profile.graph;const auto width=profile.width;
  Result out;out.continuation=export_continuation();
  for(const auto& atom:download_atoms(outputs->atoms()))out.outputs.push_back({atom.batch,atom.time,atom.source,atom.value});
  std::sort(out.outputs.begin(),out.outputs.end(),[&](const Output& a,const Output& b){return
    std::tie(a.time,a.batch,g.outputs[a.port],a.port)<std::tie(b.time,b.batch,g.outputs[b.port],b.port);});
  out.stats={{"device_stages",stages.cpu().item<Index>()},{"events",event_count.cpu().item<Index>()},
    {"pending_peak",pending->stats().cpu()[1].item<Index>()},{"prefill",limits.prefill},{"diagnostics",limits.diagnostics},
    {"full_chunks",full->chunks().cpu().item<Index>()},{"full_chunk_rows",full->chunk_rows()},
    {"lh_full_chunk_rows",lh_full?lh_full->chunk_rows():0},
    {"swiglu_full_chunk_rows",swiglu_full?swiglu_full->chunk_rows():0},
    {"attention_chunks",attention?attention->chunks().cpu().item<Index>():0},
    {"attention_chunk_rows",attention?attention->chunk_rows():0},
    {"attention_kv_peak",attention?attention->peak().cpu().item<Index>():0},
    {"attention_kv_capacity",attention?limits.kv_rows:0},
    {"event_attention_chunks",event_attention?event_attention->chunks().cpu().item<Index>():0},
    {"event_attention_chunk_rows",event_attention?event_attention->chunk_rows():0},
    {"event_attention_kv_peak",event_attention?event_attention->peak().cpu().item<Index>():0},
    {"event_attention_kv_capacity",event_attention?limits.kv_rows:0},
    {"emission_chunks",emission->chunks().cpu().item<Index>()},{"emission_chunk_rows",emission->chunk_rows()}};
  out.stats["memory_budget_bytes"]=limits.workspace_bytes;
  out.stats["usable_memory_budget_bytes"]=usable_memory_budget;
  out.stats["planned_buffer_bytes"]=planned_buffer_bytes;
  out.stats["cann_workspace_budget_bytes"]=operator_workspace_budget;
  out.stats["cann_workspace_bytes"]=program->workspace_bytes();
  out.stats["retained_tensor_bytes"]=program->retained_tensor_bytes();
  out.stats["planned_headroom_bytes"]=limits.workspace_bytes-planned_buffer_bytes-program->workspace_bytes();
  out.stats["aggressive_chunking"]=limits.chunk_policy==ChunkPolicy::aggressive;
  for(const bool event:{false,true}) {
    const auto work=event?(event_attention?event_attention->key_work():at::zeros({3},at::kLong)):
      (attention?attention->key_work().cpu():at::zeros({3},at::kLong));
    const std::string prefix=event?"event_attention_":"attention_";
    out.stats[prefix+"key_rows"]=event?(event_attention?event_attention->key_rows():0):(attention?attention->key_rows():0);
    out.stats[prefix+"key_tiles"]=work[0].item<Index>();
    out.stats[prefix+"tiled_score_entries"]=work[1].item<Index>();
    out.stats[prefix+"tiled_padding_entries"]=work[2].item<Index>();
  }
  if(!limits.diagnostics)return out;
  out.messages=download_atoms(messages->atoms());
  std::sort(out.messages.begin(),out.messages.end(),[&](const Atom& a,const Atom& b){return
    std::tie(a.position,a.batch,g.edges[a.source].source,a.source)<std::tie(b.position,b.batch,g.edges[b.source].source,b.source);});
  const auto n=events->count.cpu().item<Index>(),na=fibers->count.cpu().item<Index>();
  if(n!=out.stats.at("events"))throw std::logic_error("incomplete content event journal");
  auto em=events->meta.cpu(),ev=events->values.cpu(),fm=fibers->meta.cpu(),fv=fibers->values.cpu(),cv=contributions->values.cpu();
  auto full_values=full_trace->values.cpu();
  auto meta=em.accessor<Index,2>(),atoms=fm.accessor<Index,2>();
  using Key=std::array<Index,3>;
  std::map<Key,std::vector<Index>> by_fiber;
  std::map<Key,std::vector<SlotValue>> emitted;
  const auto ne=emission_trace->count.cpu().item<Index>();
  auto sm=emission_trace->meta.cpu(),sv=emission_trace->values.cpu();auto slots=sm.accessor<Index,2>();
  for(Index i=0;i<ne;++i)emitted[{slots[i][0],slots[i][1],slots[i][2]}].push_back({slots[i][4],sv[i].clone()});
  std::map<Key,Index> node_batches;Index max_batch=0,max_causal=0,max_state_read=0;
  for(Index i=0;i<na;++i)by_fiber[{atoms[i][0],atoms[i][1],atoms[i][2]}].push_back(i);
  for(Index i=0;i<n;++i) {
    const auto size=++node_batches[{meta[i][12],meta[i][0],meta[i][1]}],region=g.nodes[meta[i][1]].region;
    max_batch=std::max(max_batch,size);
    if(profile.causal_regions[region])max_causal=std::max(max_causal,size);
    if(g.regions[region].read_mode!="content")max_state_read=std::max(max_state_read,size);
    Event e;e.batch=meta[i][0];e.node=meta[i][1];e.time=meta[i][2];e.active=meta[i][3];
    e.content=ev[i].narrow(0,0,width).clone();
    e.old={ev[i].narrow(0,width,width).clone(),meta[i][4],meta[i][5]};
    e.proposed_state={ev[i].narrow(0,2*width,width).clone(),meta[i][6],meta[i][7]};e.proposal=e.proposed_state.value;
    e.comparison_state={ev[i].narrow(0,3*width,width).clone(),meta[i][8],meta[i][9]};e.comparison=e.comparison_state.value;
    e.next_state={ev[i].narrow(0,4*width,width).clone(),meta[i][10],meta[i][11]};e.next=e.next_state.value;
    e.descriptor=ev[i][5*width].clone();e.control=ev[i][5*width+1].clone();
    if(e.active){e.full=full_values[i].clone();e.emitted=std::move(emitted[{e.batch,e.node,e.time}]);}
    for(auto row:by_fiber.at({e.batch,e.node,e.time})) {
      auto c=atoms[row];Atom a{c[0],c[1],c[2],c[3],c[4],c[5],fv[row].clone()};e.fiber.push_back(a);
      Index slot=a.kind==0?g.source_domain->input[a.source]:g.source_domain->edge_target[a.source];
      const auto scale=a.kind==0?profile.model.input_scale[a.source]:profile.model.agg_scale[a.source];
      if(a.kind==1&&!g.origins.empty()&&g.origin_index[a.source]>=0) {
        const auto& origin=g.origins[g.origin_index[a.source]];
        if(a.position%origin.stride)throw std::logic_error("device accepted off-lattice input origin");
        a.kind=0;a.source=origin.port;a.position/=origin.stride;
      }
      e.sources.push_back({slot,a,scale});
      e.contributions.push_back({slot,cv[row].clone()});
    }
    if(!g.origins.empty())std::stable_sort(e.sources.begin(),e.sources.end(),[](const SourceInput& a,const SourceInput& b){return a.atom.key()<b.atom.key();});
    std::sort(e.contributions.begin(),e.contributions.end(),[](const SlotValue& a,const SlotValue& b){return a.slot<b.slot;});
    out.trace.push_back(std::move(e));
  }
  if(attention)attention->export_trace(out.trace);
  if(event_attention)event_attention->export_trace(out.trace);
  // Trace histories are materialized from recorded device active bits and the
  // starting history. Final history above is independently read from the NPU.
  std::map<Key,std::vector<size_t>> frames;
  for(size_t i=0;i<out.trace.size();++i){const auto& e=out.trace[i];frames[{e.batch,g.nodes[e.node].region,e.time}].push_back(i);}
  auto histories=download_history(g,history_before,boundary.batch_size);
  for(const auto& [key,ids]:frames) {
    auto& h=histories[{key[0],key[1]}];auto& selections=h.node_maps["selected"];h.last_time=key[2];
    for(auto i:ids)if(out.trace[i].active)++selections[out.trace[i].node];
    for(auto i:ids)out.trace[i].history=h;
  }
  std::sort(out.trace.begin(),out.trace.end(),[](const Event& a,const Event& b){return std::tie(a.time,a.batch,a.node)<std::tie(b.time,b.batch,b.node);});
  out.stats.insert({{"messages",Index(out.messages.size())},{"max_node_time_batch",max_batch},
    {"state_read_single_frame_regions",std::count(profile.causal_regions.begin(),profile.causal_regions.end(),1)},
    {"max_causal_node_time_batch",max_causal},{"max_state_read_node_time_batch",max_state_read}});
  return out;
}
} // namespace tide::device_online
