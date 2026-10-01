#include "resident_consumer.h"
#include <ATen/core/grad_mode.h>
#include <stdexcept>
namespace tide_flow {
tide::ResidentTrainingLimits resident_limits(const Config& c,at::Device device) {
  tide::ResidentTrainingLimits out;auto& f=out.forward;
  f.workspace_bytes=512LL*1024*1024;f.prefill=c.schedule=="prefill";f.diagnostics=c.training||c.diagnostics;
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
  if(c.devices>1)for(Index i=0;i<c.devices;++i)out.placement.devices.emplace_back(device.type(),device.index()+i);
  out.placement.policy=c.owner_policy;return out;
}
std::string run_resident(const Packet& p,const Config& c,at::Device device,std::ostream* diagnostics) {
  if(device.type()!=c10::DeviceType::PrivateUse1||device.index()<0||c.runtime.dtype!=at::kFloat)
    throw std::invalid_argument("resident consumer requires explicit logical NPU FP32");
  if(!c.training&&c.devices>1)throw std::invalid_argument("multi-device resident inference consumer pending; no training-tape substitution");
  auto placement=tide::resolve_placement(c.placement,device);
  if(placement.read!=device||placement.control!=device||placement.selection!=device||placement.events!=device
      ||placement.scoring_dtype=="float64")throw std::invalid_argument("resident consumer requires all phases on NPU with FP32 scoring");
  at::NoGradGuard no_grad;auto begin=Clock::now();auto f=fixture(p,c,at::Device(at::kCPU));
  // The fixture installs eager built-in handles during validation/embedding.
  // Resident reconstructs these same declared modules from graph metadata.
  for(auto& w:f.model.nodes){w.kernel.reset();w.read_kernel.reset();w.next_kernel.reset();w.aggregate_kernel.reset();w.full_kernel.reset();}
  for(auto& w:f.model.regions)w.kernel.reset();
  auto embedding=f.embedding.detach().to(device),head=f.head.detach().to(device);
  tide::Continuation q;q.identity=f.graph.identity;q.batch_size=p.batch;
  ResidentMeasurements result;result.limits=resident_limits(c,device);
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
    inference=std::make_unique<tide::ResidentSession>(f.graph,f.model,q,device,result.limits.forward);
    result.placement.devices={device};result.placement.policy=c.owner_policy;
  }
  auto sync=[&]{for(const auto& d:result.placement.devices)portable_torch::synchronize(d);};
  sync();result.construction=seconds(begin);Index position=0;
  for(Index step=0;step<c.steps+c.warmup;++step) {
    sync();begin=Clock::now();Tensor loss,gh;Index count=0;std::vector<Tensor> counters;
    std::vector<tide::ResidentCotangents> roots;std::map<std::string,Index> reverse_statistics;
    for(Index window=0;window<c.windows;++window) {
      auto positions=at::arange(position,position+p.tokens,at::kLong).reshape({1,-1});
      auto samples=at::arange(p.batch,at::kLong).reshape({-1,1});
      auto ids=(positions*7+samples*3).remainder(p.vocab).to(device);
      auto values=at::embedding(embedding,ids);std::vector<tide::External> external;
      for(Index b=0;b<p.batch;++b)for(Index t=0;t<p.tokens;++t)
        external.push_back({b,0,position+t,(position+t)*p.stride,values[b][t]});
      const auto stop=(position+p.tokens)*p.stride;tide::ResidentTrainingWindow retained;
      tide::ResidentWindow output;
      if(training){retained=training->advance(external,stop,stop);output=retained.outputs;}
      else output=inference->advance(external,stop,stop);
      auto item=head_loss(output,head,p,p.batch*p.tokens*c.windows,c.training);
      if(item.value.defined())loss=loss.defined()?loss+item.value:item.value;
      if(item.head_gradient.defined())gh=gh.defined()?gh+item.head_gradient:item.head_gradient;
      if(training) {
        tide::ResidentCotangents root;root.token=retained.token;root.outputs=item.root;
        if(item.root.defined())root.outputs_connected=output.valid;roots.push_back(root);
      }
      count+=item.count;position+=p.tokens;
      counters.push_back(at::stack({output.stages.reshape({}),output.events.reshape({}),
                                   output.full_chunks.reshape({}),output.emission_chunks.reshape({})}).clone());
      if(diagnostics) {
        auto value=training?training->result():inference->result();if(c.family=="settle")value=f.settle->project(value);
        window_json(*diagnostics,step,value);
      }
    }
    if(loss.defined()&&!at::isfinite(loss).all().item<bool>())throw std::runtime_error("nonfinite consumer loss; optimizer not applied");
    if(training) {
      const auto gradient=training->backward(roots);reverse_statistics=gradient.statistics;auto ge=embedding_gradient(gradient,embedding);
      optimizer->prepare(ge,gh);
      if(diagnostics)resident_gradients_json(*diagnostics,step,f,gradient,ge,gh);
      const auto accepted=training->step();
      if(!accepted.applied)throw std::runtime_error("graph optimizer refused; consumer parameters unchanged, code="+std::to_string(accepted.refusal_code));
      optimizer->commit();
    }
    sync();const auto elapsed=seconds(begin);
    if(diagnostics) {
      auto checkpoint=training?training->checkpoint():tide::ResidentTrainingCheckpoint{};
      resident_updated_json(*diagnostics,step,f,training?&checkpoint:nullptr,embedding,head);
    }
    if(step>=c.warmup) {
      result.seconds.push_back(elapsed);result.losses.push_back(loss.defined()?loss.cpu().item<double>():0.);result.outputs.push_back(count);
      auto counts=at::stack(counters).sum(0).cpu().contiguous();auto values=counts.data_ptr<Index>();
      result.statistics.push_back({{"stages",values[0]},{"events",values[1]},{"full_chunks",values[2]},{"emission_chunks",values[3]}});
      result.statistics.back().insert(reverse_statistics.begin(),reverse_statistics.end());
    }else result.warmup.push_back(elapsed);
  }
  result.cut=training?training->cut():inference->cut();if(training)training->close();else inference->close();
  return resident_record(p,c,device,result);
}
} // namespace tide_flow
