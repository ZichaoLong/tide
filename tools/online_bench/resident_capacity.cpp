#include "resident_consumer.h"
#include "memory.h"
#include "capacity_record.h"
namespace tide_flow {
void prepare_capacity(const Packet& p,const Config& c,const std::vector<at::Device>& devices,ResidentMeasurements& result) {
  const Index n=p.body.nodes.size();const auto& f=result.limits.forward;
  result.sample_rows=std::min(c.sample_chunk_rows?c.sample_chunk_rows:p.batch,p.batch);
  result.sample_chunks=(p.batch-1)/result.sample_rows+1;
  capacity::Geometry g{p.width,result.sample_rows,p.vocab,c.windows,c.runtime.dtype==at::kHalf?2:4,p.memory=="attention",
    c.training,c.optimizer=="adamw",c.training||f.diagnostics,Index(p.body.regions.size()+2),std::vector<Index>(n),std::vector<Index>(n),{},c.devices,c.owner_policy=="locality",result.sample_chunks};
  for(const auto& e:p.body.edges){++g.sources[e.target];++g.slots[e.source];g.edges.emplace_back(e.source,e.target);}
  for(auto node:p.body.inputs){++g.sources[node];g.edges.emplace_back(n,node);}
  for(auto node:p.body.outputs){++g.slots[node];g.edges.emplace_back(node,n+1);}
  const capacity::Capacities caps{f.queue,f.arrivals,f.outputs,f.trace,f.kv_rows,f.kv_trace_rows,result.limits.program_workspace_bytes};
  const capacity::Chunks chunks{{"full",f.full_chunk_rows},{"emission",f.emission_chunk_rows},{"aggregate",f.aggregate_chunk_rows},
    {"attention",f.attention_chunk_rows},{"keys",f.attention_key_rows},{"reverse",result.limits.reverse_chunk_rows},{"head",result.head.rows}};
  std::vector<Index> budgets;
  for(auto d:devices){auto value=device_memory_info(d);result.initial_memory.push_back(value);budgets.push_back(c.device_memory_bytes?std::min(value.free,c.device_memory_bytes):value.free);}
  result.capacity=capacity::plan(g,caps,chunks,budgets,c.chunk_policy=="aggressive");
  capacity::Wide snapshot=0,accumulation=0;
  for(const auto& card:result.capacity.cards) {
    snapshot+=card.components.at("continuation_snapshot_bytes");
    accumulation+=card.components.at("gradient_accumulation");
  }
  result.snapshot_budget=capacity::bytes(snapshot);result.accumulation_budget=capacity::bytes(accumulation);
  auto& forward=result.limits.forward;const auto& selected=result.capacity.effective;
  forward.full_chunk_rows=selected.at("full");forward.emission_chunk_rows=selected.at("emission");forward.aggregate_chunk_rows=selected.at("aggregate");
  forward.attention_chunk_rows=selected.at("attention");forward.attention_key_rows=selected.at("keys");result.limits.reverse_chunk_rows=selected.at("reverse");
  result.head.rows=selected.at("head");result.head.reserved_bytes=result.head.fixed_bytes+result.head.rows*result.head.row_bytes;
  if(!result.limits.placement.devices.empty())result.limits.placement.full_owners=result.limits.placement.state_owners=result.capacity.owners;
}
std::string capacity_json(const Config& c,const ResidentMeasurements& r) {
  std::ostringstream base;capacity::record(base,r.capacity);auto value=base.str();value.pop_back();
  std::ostringstream out;out<<value<<",\"requested_device_memory_bytes\":"<<c.device_memory_bytes<<",\"initial_devices\":[";
  for(size_t i=0;i<r.initial_memory.size();++i){if(i)out<<',';const auto& d=r.initial_memory[i];
    out<<"{\"device\":"<<quoted(d.device.str())<<",\"free_bytes\":"<<d.free<<",\"total_bytes\":"<<d.total<<",\"allocated_bytes\":"<<d.allocated<<'}';}
  out<<"],\"observed_peak_growth_bytes\":";capacity::array(out,r.peak_growth);out<<",\"allocator_within_estimate\":true}";return out.str();
}
} // namespace tide_flow
