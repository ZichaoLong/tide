#include "resident_consumer.h"
#include <iomanip>
namespace tide_flow {
namespace {
void parameters(std::ostream& out,Index step,const std::map<std::string,Tensor>& values,bool gradients) {
  out<<"{\"kind\":"<<quoted(gradients?"gradients":"updated")<<",\"step\":"<<step<<",\"parameters\":{";
  bool first=true;for(const auto& [name,value]:values){if(!first)out<<',';first=false;out<<quoted(name)<<':';tensor_json(out,value);}out<<"}}\n";
}
}
void resident_gradient_add(std::map<std::string,Tensor>& values,const Fixture& f,const tide::ResidentGradients& g) {
  for(const auto& owner:f.parameters.owners())values.emplace(owner.canonical,Tensor());
  auto append=[&](const auto& shard) {
    auto flags=shard.connected.cpu(),data=shard.values.cpu();
    for(size_t i=0;i<shard.names.size();++i)if(flags[i].template item<bool>()) {
      const auto shape=f.parameters.value(shard.names[i]);
      const auto part=data.narrow(0,shard.offsets[i],shape.numel()).reshape(shape.sizes());
      auto& target=values[shard.names[i]];target=target.defined()?target+part:part;
    }
  };
  if(g.parameter_shards.empty())append(g);else for(const auto& s:g.parameter_shards)append(s);
}
void resident_gradients_json(std::ostream& out,Index step,std::map<std::string,Tensor> values,const Tensor& ge,const Tensor& gh) {
  values["embedding"]=ge;values["head"]=gh;parameters(out,step,values,true);
}
void resident_updated_json(std::ostream& out,Index step,const Fixture& f,const tide::ResidentTrainingCheckpoint* saved,const Tensor& embedding,const Tensor& head) {
  std::map<std::string,Tensor> values;
  for(const auto& owner:f.model.parameters(true).owners())values[owner.canonical]=saved?saved->parameters.at(owner.canonical):owner.value;
  values["embedding"]=embedding;values["head"]=head;parameters(out,step,values,false);
}
std::string resident_record(const Packet& p,const Config& c,at::Device device,const ResidentMeasurements& r) {
  std::ostringstream out;out<<std::setprecision(17);
  out<<"{\"schema\":\"tide-online-consumer-v1\",\"state\":\"passed\",\"workload_sha256\":"<<quoted(p.sha)
     <<",\"packet_identity\":\"declared; launcher must verify v2 text against hashed JSON\",\"implementation\":\"libtorch\",\"family\":"<<quoted(c.family)
     <<",\"training\":"<<(c.training?"true":"false")<<",\"optimizer\":"<<(c.training?quoted(c.optimizer):"null")
     <<",\"parameters\":"<<p.parameters()<<",\"construction_seconds\":"<<r.construction;
  auto array=[&](const std::string& name,const auto& values){out<<','<<quoted(name)<<":[";bool first=true;for(const auto v:values){if(!first)out<<',';first=false;out<<v;}out<<']';};
  array("seconds",r.seconds);array("warmup_seconds",r.warmup);array("outputs",r.outputs);
  out<<",\"losses\":[";for(size_t i=0;i<r.losses.size();++i){if(i)out<<',';if(r.outputs[i])out<<r.losses[i];else out<<"null";}out<<']';
  out<<",\"statistics\":[";bool first=true;for(const auto& s:r.statistics){if(!first)out<<',';first=false;out<<'{';bool field=true;
    for(const auto& [k,v]:s){if(!field)out<<',';field=false;out<<quoted(k)<<':'<<v;}out<<'}';}out<<']';
  out<<",\"windows_per_step\":"<<c.windows<<",\"warmup_steps\":"<<c.warmup<<",\"measured_steps\":"<<c.steps
     <<",\"input_tokens_per_step\":"<<p.batch*p.tokens*c.windows<<",\"final_cut\":"<<r.cut
     <<",\"batch_execution\":{\"logical_batch\":"<<p.batch<<",\"requested_sample_chunk_rows\":"<<c.sample_chunk_rows
     <<",\"effective_sample_chunk_rows\":"<<r.sample_rows<<",\"physical_chunks\":"<<r.sample_chunks<<'}'
     <<",\"threads\":"<<c.threads<<",\"parameter_budget\":"<<c.parameter_budget<<",\"diagnostics\":"<<(c.diagnostics?"true":"false")
     <<",\"precision\":{\"payload\":"<<quoted(portable_torch::dtype_name(c.runtime.dtype))
     <<",\"loss\":\"float32\",\"adjoints\":\"float32\",\"optimizer_masters\":\"float32\"}"
     <<",\"head_memory\":{\"rows\":"<<r.head.rows<<",\"fixed_bytes\":"<<r.head.fixed_bytes
     <<",\"row_bytes\":"<<r.head.row_bytes<<",\"reserved_bytes\":"<<r.head.reserved_bytes<<",\"budget\":"<<r.head.budget
     <<",\"operator_allowance_bytes\":"<<r.head.operator_allowance_bytes<<'}'
     <<",\"memory\":"<<r.memory
     <<",\"memory_admission\":"<<capacity_json(c,r)
     <<",\"runtime\":{\"device\":"<<quoted(device.str())<<",\"dtype\":"<<quoted(portable_torch::dtype_name(c.runtime.dtype))
     <<",\"backend\":"<<quoted(portable_torch::compiled_backend())<<",\"resolution_reason\":"<<quoted(portable_torch::resolution_reason(c.runtime,device))
     <<",\"schedule\":"<<quoted(c.schedule)<<",\"preset\":\"resident\",\"placement\":{";
  first=true;for(const auto& [k,v]:tide::resolve_placement(c.placement,device).record()){if(!first)out<<',';first=false;out<<quoted(k)<<':'<<quoted(v);}
  out<<"},\"resident\":{\"runtime_owner\":\"standalone\",\"requested_devices\":"<<c.devices
     <<",\"requested_owner_policy\":"<<quoted(c.owner_policy)<<",\"devices\":[";
  first=true;for(const auto& d:r.placement.devices){if(!first)out<<',';first=false;out<<quoted(d.str());}out<<']';
  array("full_owners",r.placement.full_owners);array("state_owners",r.placement.state_owners);
  const auto& f=r.limits.forward;
  const std::map<std::string,Index> limits={{"queue",f.queue},{"arrivals",f.arrivals},{"outputs",f.outputs},{"trace",f.trace},{"stages",f.stages},
    {"workspace_bytes",f.workspace_bytes},{"full_chunk_rows",f.full_chunk_rows},{"emission_chunk_rows",f.emission_chunk_rows},
    {"aggregate_chunk_rows",f.aggregate_chunk_rows},{"attention_chunk_rows",f.attention_chunk_rows},{"attention_key_rows",f.attention_key_rows},
    {"kv_rows",f.kv_rows},{"kv_trace_rows",f.kv_trace_rows},{"max_repeat_ticks",f.max_repeat_ticks},{"windows",r.limits.windows},
    {"retained_bytes",r.limits.retained_bytes},{"backward_bytes",r.limits.backward_bytes},{"optimizer_bytes",r.limits.optimizer_bytes},
    {"program_workspace_bytes",r.limits.program_workspace_bytes},{"reverse_chunk_rows",r.limits.reverse_chunk_rows}};
  out<<",\"owner_policy\":"<<quoted(r.placement.policy)<<",\"chunk_policy\":"<<quoted(c.chunk_policy)<<",\"limits\":{";first=true;
  for(const auto& [k,v]:limits){if(!first)out<<',';first=false;out<<quoted(k)<<':'<<v;}out<<"}}}"
     <<",\"timing\":\"input preparation/upload + online resident graph + packed output head/loss + "
     <<(c.training?"graph/input/embedding VJP + finite staged optimizer + ":"finite loss check + ")
     <<"synchronization; no reference\""
     <<",\"boundary_policy\":\"dynamic output compaction at window boundary; scheduling remains device-owned; external input metadata prepared on host\""
     <<",\"projection_placement\":"<<quoted(r.limits.placement.devices.empty()?"coordinator dense":"compact banks on Full owners")<<"}\n";
  return out.str();
}
} // namespace tide_flow
