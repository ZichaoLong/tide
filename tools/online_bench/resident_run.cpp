#include "resident_consumer.h"
#include "memory.h"
#include "resident_contexts.h"
#include "failure.h"
#include <ATen/core/grad_mode.h>
#include <stdexcept>
namespace tide_flow {
tide::ResidentTrainingLimits resident_limits(const Config& c,at::Device device) {
  tide::ResidentTrainingLimits out;auto& f=out.forward;
  f.workspace_bytes=512LL*1024*1024;f.prefill=c.schedule=="prefill";f.diagnostics=c.diagnostics;
  f.chunk_policy=c.chunk_policy=="aggressive"?tide::ResidentChunkPolicy::aggressive:tide::ResidentChunkPolicy::conservative;
  out.windows=c.windows;out.backward_bytes=2LL*1024*1024*1024;
  std::map<std::string,Index*> fields={{"queue",&f.queue},{"arrivals",&f.arrivals},{"outputs",&f.outputs},
    {"trace",&f.trace},{"stages",&f.stages},{"workspace-bytes",&f.workspace_bytes},
    {"full-chunk-rows",&f.full_chunk_rows},{"emission-chunk-rows",&f.emission_chunk_rows},{"aggregate-chunk-rows",&f.aggregate_chunk_rows},
    {"attention-chunk-rows",&f.attention_chunk_rows},{"attention-key-rows",&f.attention_key_rows},
    {"kv-rows",&f.kv_rows},{"kv-trace-rows",&f.kv_trace_rows},{"max-repeat-ticks",&f.max_repeat_ticks},
    {"retained-bytes",&out.retained_bytes},{"backward-bytes",&out.backward_bytes},{"optimizer-bytes",&out.optimizer_bytes},
    {"program-workspace-bytes",&out.program_workspace_bytes},{"reverse-chunk-rows",&out.reverse_chunk_rows}};
  for(const auto& [name,value]:c.resident_limits) {
    if(value<1&&name!="trace")throw std::invalid_argument("resident limits must be positive");
    *fields.at(name)=value;
  }
  if(c.devices>1||!c.owner_map.empty())for(Index i=0;i<c.devices;++i)out.placement.devices.emplace_back(device.type(),device.index()+i);
  out.placement.policy=c.owner_policy;return out;
}
std::string run_resident(const Packet& p,const Config& c,at::Device device,std::ostream* diagnostics) {
  if(device.type()!=c10::DeviceType::PrivateUse1||device.index()<0||(c.runtime.dtype!=at::kFloat&&c.runtime.dtype!=at::kHalf))
    throw std::invalid_argument("resident consumer requires explicit logical NPU FP32/FP16");
  auto placement=tide::resolve_placement(c.placement,device);
  if(placement.read!=device||placement.control!=device||placement.selection!=device||placement.events!=device
      ||placement.scoring_dtype=="float64")throw std::invalid_argument("resident consumer requires all phases on NPU with FP32 scoring");
  at::NoGradGuard no_grad;auto begin=Clock::now();
  std::vector<at::Device> memory_devices;
  for(Index i=0;i<c.devices;++i)memory_devices.emplace_back(device.type(),device.index()+i);
  MemoryRecord memory(memory_devices);
  ResidentMeasurements result;result.limits=resident_limits(c,device);result.phases.enabled=c.phase_timing;
  result.head=head_budget(result.limits.forward.outputs,p.width,p.vocab,c.runtime.dtype==at::kHalf?2:4,
    c.training,c.head_workspace_bytes,c.chunk_policy=="aggressive");
  prepare_capacity(p,c,memory_devices,result);
  auto f=fixture(p,c,at::Device(at::kCPU));
  // The fixture installs eager built-in handles during validation/embedding.
  // Resident reconstructs these same declared modules from graph metadata.
  for(auto& w:f.model.nodes){w.kernel.reset();w.read_kernel.reset();w.next_kernel.reset();w.aggregate_kernel.reset();w.full_kernel.reset();}
  for(auto& w:f.model.regions)w.kernel.reset();
  auto embedding=f.embedding.detach().to(device),head=f.head.detach().to(device);
  tide::Continuation q;q.identity=f.graph.identity;q.batch_size=result.sample_rows;
  std::unique_ptr<tide::ResidentTrainingSession> training;
  std::unique_ptr<tide::ResidentSession> inference;
  std::unique_ptr<ConsumerOptimizer> optimizer;
  if(c.training) {
    tide::OptimizerGroup group;group.lr=.0001;group.weight_decay=.001;group.momentum=.25;group.eps=1e-6;
    for(const auto& owner:f.model.parameters(true).owners())group.parameters.push_back(owner.canonical);
    training=std::make_unique<tide::ResidentTrainingSession>(f.graph,f.model,q,device,
      c.optimizer=="sgd"?tide::ResidentOptimizerKind::sgd:tide::ResidentOptimizerKind::adamw,std::vector<tide::OptimizerGroup>{group},result.limits);
    result.placement=training->placement();optimizer=std::make_unique<ConsumerOptimizer>(embedding,head,c.optimizer);
  } else {
    inference=std::make_unique<tide::ResidentSession>(f.graph,f.model,q,device,result.limits.forward,result.limits.placement);
    result.placement=inference->placement();
  }
  auto sync=[&]{for(const auto& d:result.placement.devices)portable_torch::synchronize(d);};
  ContextPool contexts(result.sample_chunks,memory_devices,result.capacity);
  auto save_context=[&] {
    const auto budgets=contexts.remaining();Index total=0;
    for(const auto& [_,value]:budgets)total=capacity::bytes(capacity::Wide(total)+value);
    if(total<1)throw std::invalid_argument("saved continuation pool budget exhausted");
    return training?training->snapshot_device(total,c.context_memory_bytes>0,budgets)
                   :inference->snapshot_device(total,c.context_memory_bytes>0,budgets);
  };
  if(!contexts.empty())contexts.initialize(save_context());
  sync();result.construction=seconds(begin);Index position=0;
  memory.capture("construction");
  for(Index step=0;step<c.steps+c.warmup;++step) {
    sync();begin=Clock::now();Tensor loss,ge,gh;Index count=0,head_chunks=0;std::vector<Tensor> counters;
    std::map<std::string,Index> reverse_statistics;std::map<std::string,Tensor> diagnostic_gradients;
    for(Index index=0,first=0;first<p.batch;++index,first+=result.sample_rows) {
      const auto size=std::min(result.sample_rows,p.batch-first);
      if(!contexts.empty()) {
        if(training)training->restore_device(contexts[index]);else inference->restore_device(contexts[index]);
        contexts.release(index);
      }
      std::vector<tide::ResidentCotangents> roots;
      for(Index window=0;window<c.windows;++window) {
        const auto cursor=position+window*p.tokens;
        auto positions=at::arange(cursor,cursor+p.tokens,at::kLong).reshape({1,-1});
        auto samples=at::arange(first,first+size,at::kLong).reshape({-1,1});
        auto ids=(positions*7+samples*3).remainder(p.vocab).to(device);
        auto values=at::embedding(embedding,ids);std::vector<tide::External> external;
        // Unused tail capacity receives no inputs, not zero-valued messages.
        for(Index b=0;b<size;++b)for(Index t=0;t<p.tokens;++t)
          external.push_back({b,0,cursor+t,(cursor+t)*p.stride,values[b][t]});
        const auto stop=(cursor+p.tokens)*p.stride;tide::ResidentTrainingWindow retained;
        tide::ResidentWindow output;
        if(training){retained=training->advance(external,stop,stop);output=retained.outputs;}
        else output=inference->advance(external,stop,stop);
        auto item=head_loss(output,head,p,p.batch*p.tokens*c.windows,c.training,result.head,first);head_chunks+=item.chunks;
        if(item.value.defined())loss=loss.defined()?loss+item.value:item.value;
        if(item.head_gradient.defined()){if(gh.defined())gh.add_(item.head_gradient);else gh=item.head_gradient;}
        if(training) {
          tide::ResidentCotangents root;root.token=retained.token;root.outputs=item.root;
          if(item.root.defined())root.outputs_connected=output.valid;roots.push_back(root);
        }
        count+=item.count;
        counters.push_back(at::stack({output.stages.reshape({}),output.events.reshape({}),
                                     output.full_chunks.reshape({}),output.emission_chunks.reshape({}),
                                     output.pending_stats[1],output.output_stats[1]}).clone());
        if(diagnostics) {
          auto value=training?training->result():inference->result();if(c.family=="settle")value=f.settle->project(value);
          value.continuation.batch_size=size;
          window_json(*diagnostics,step,value,first,contexts.empty()?0:p.batch);
        }
      }
      if(training) {
        const auto gradient=training->backward(roots);
        for(const auto& [key,value]:gradient.statistics) {
          auto& total=reverse_statistics[key];
          total=key.size()>=5&&key.substr(key.size()-5)=="bytes"?std::max(total,value):total+value;
        }
        auto partial=embedding_gradient(gradient,embedding,first);
        if(partial.defined()){if(ge.defined())ge.add_(partial);else ge=partial;}
        if(diagnostics)resident_gradient_add(diagnostic_gradients,f,gradient);
        if(!contexts.empty())training->accumulate(result.accumulation_budget);
      }
      roots.clear();
      if(!contexts.empty())contexts.store(index,save_context());
    }
    position+=c.windows*p.tokens;
    double sample_seconds=-1;
    if(c.phase_timing&&training){sync();sample_seconds=seconds(begin);}
    if(loss.defined()&&!at::isfinite(loss).all().item<bool>())throw std::runtime_error("nonfinite consumer loss; optimizer not applied");
    if(training) {
      optimizer->prepare(ge,gh);
      if(diagnostics)resident_gradients_json(*diagnostics,step,std::move(diagnostic_gradients),ge,gh);
      const auto accepted=training->step();
      if(!accepted.applied)throw std::runtime_error("graph optimizer refused; consumer parameters unchanged, code="+std::to_string(accepted.refusal_code));
      optimizer->commit();
    }
    sync();const auto elapsed=seconds(begin);
    result.phases.add(elapsed,sample_seconds,step<c.warmup);
    if(diagnostics) {
      auto checkpoint=training?training->checkpoint():tide::ResidentTrainingCheckpoint{};
      resident_updated_json(*diagnostics,step,f,training?&checkpoint:nullptr,embedding,head);
    }
    if(step>=c.warmup) {
      result.seconds.push_back(elapsed);result.losses.push_back(loss.defined()?loss.cpu().item<double>():0.);result.outputs.push_back(count);
      const auto rows=at::stack(counters);
      // Preserve int64 counters; one boundary transfer for totals and peaks.
      auto counts=at::cat({rows.sum(0).narrow(0,0,4),std::get<0>(rows.max(0))}).cpu().contiguous();
      auto values=counts.data_ptr<Index>();
      result.statistics.push_back({{"stages",values[0]},{"events",values[1]},{"full_chunks",values[2]},{"emission_chunks",values[3]},
        {"window_stages_max",values[4]},{"window_events_max",values[5]},{"pending_peak",values[8]},{"window_outputs_max",values[9]}});
      result.statistics.back().insert(reverse_statistics.begin(),reverse_statistics.end());
      result.statistics.back()["head_chunks"]=head_chunks;
    }else result.warmup.push_back(elapsed);
    if(step+1==c.warmup)memory.capture("warmup");
  }
  memory.capture("measured",false);result.memory=memory.json();
  result.peak_growth=memory.peak_growth();
  result.context_peaks=contexts.peaks();
  result.cut=training?training->cut():inference->cut();if(training)training->close();else inference->close();
  if(!capacity::within_estimate(result.capacity,result.peak_growth)) {
    const std::string error="consumer memory estimate underestimated allocator peak; retain failed run and recalibrate";
    throw RecordedFailure(error,resident_record(p,c,device,result,error));
  }
  return resident_record(p,c,device,result);
}
} // namespace tide_flow
