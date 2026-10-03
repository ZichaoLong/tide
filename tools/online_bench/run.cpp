#include "consumer.h"
#include "phase_timing.h"
#include "memory.h"
#include "eager_placement.h"
#include "eager_capacity_runtime.h"
#include "eager_optimizer.h"
#include <tide/greedy.h>
#include <tide/stream.h>
#include <ATen/core/grad_mode.h>
#include <torch/csrc/autograd/autograd.h>
#include <iomanip>
#include <limits>
#include <stdexcept>

namespace tide_flow {
namespace {
void detach(tide::Continuation& q) {
  for(auto& [_,s]:q.states){s.value=s.value.detach();for(auto& [__,v]:s.slots)v=v.detach();}
  for(auto& [_,h]:q.history)for(auto& [__,v]:h.tensors)v=v.detach();
  for(auto& a:q.pending)a.value=a.value.detach();
}
Tensor objective(const tide::Result& r,const Fixture& f,const Packet& p,Index windows,Index sample_begin) {
  if(r.outputs.empty())return {};
  std::vector<Tensor> rows;std::vector<Index> labels;
  for(const auto& x:r.outputs){rows.push_back(x.value.to(f.head.device()));labels.push_back(((x.time/p.stride+1)*7+(x.batch+sample_begin)*3)%p.vocab);}
  auto logits=at::matmul(at::stack(rows),f.head.t());
  if(logits.scalar_type()==at::kHalf)logits=logits.to(at::kFloat);
  return at::cross_entropy_loss(logits,at::tensor(labels,at::kLong).to(f.head.device()),{},at::Reduction::Sum)
    /double(p.batch*p.tokens*windows);
}
}
std::string run(const Packet& p,const Config& c,at::Device device,std::ostream* diagnostics) {
  if(static_cast<long double>(p.parameters())*c10::elementSize(c.runtime.dtype)>c.parameter_budget)
    throw std::invalid_argument("learned parameter storage exceeds declared parameter-budget (not a total peak-memory estimate)");
  if(diagnostics&&p.parameters()>100000)throw std::invalid_argument("diagnostics require at most 100000 learned parameters");
  if(static_cast<long double>(c.steps+c.warmup)*c.windows*p.tokens*p.stride>std::numeric_limits<Index>::max()/8.L)
    throw std::invalid_argument("requested continuation would overflow logical coordinates/token formula");
  if(c.placement.preset=="resident") {
#ifdef TIDE_ONLINE_RESIDENT
    return run_resident(p,c,device,diagnostics);
#else
    throw std::invalid_argument("resident consumer requires a build with TIDE_ONLINE_RESIDENT=ON");
#endif
  }
  auto start=Clock::now();const auto owner_plan=eager_placement(p,c);const auto devices=eager_devices(device,c.devices);
  for(auto owner:devices)if(!tide::resolve_placement(c.placement,owner).events.is_cpu())
    throw std::invalid_argument("host model placement cannot provide device-resident event progression");
  auto sync=[&]{for(auto owner:devices)portable_torch::synchronize(owner);};
  eager_capacity::Admission admission(p,c,devices);
  MemoryRecord memory(devices);auto f=fixture(p,c,device);
  auto placement=tide::resolve_placement(c.placement,device);auto placed=tide::place_model(f.graph,f.model,c.placement);
  tide::Options options;options.packed=true;options.prefill=c.schedule=="prefill";options.trace=c.diagnostics;
  options.workers=c.workers;options.packed_sources=c.packed_sources;options.batch_next=c.batch_next;
  options.full_autograd="batched";options.aggregate_autograd="batched";
  std::unique_ptr<tide::Streaming> streaming;std::unique_ptr<tide::Greedy> greedy;
  if(c.schedule=="prefill")greedy=std::make_unique<tide::Greedy>(f.graph,placed,options);
  else streaming=std::make_unique<tide::Streaming>(f.graph,placed,options);
  std::unique_ptr<EagerOptimizer> optimizer;
  tide::OptimizerGroup group;group.lr=.0001;group.weight_decay=.001;group.momentum=.25;group.eps=1e-6;
  for(const auto& owner:f.parameters.owners())group.parameters.push_back(owner.canonical);
  if(c.training) {
    optimizer=std::make_unique<EagerOptimizer>(f.parameters,c.optimizer,c.runtime.dtype==at::kHalf,c.loss_scale,group);
  }
  const auto chunk=admission.rows();
  std::vector<tide::Continuation> continuations;
  for(Index first=0;first<p.batch;first+=chunk){tide::Continuation q;q.identity=f.graph.identity;
    q.batch_size=std::min(chunk,p.batch-first);continuations.push_back(std::move(q));}
  sync();const auto construction=seconds(start);
  memory.capture("construction");
  std::vector<double> durations,losses,warmup_times;std::vector<Index> outputs;std::vector<std::map<std::string,Index>> statistics;
  PhaseTiming phases{c.phase_timing};
  Index position=0;at::AutoGradMode mode(c.training);
  for(Index step=0;step<c.steps+c.warmup;++step) {
    sync();start=Clock::now();if(optimizer)optimizer->zero_grad();
    Tensor loss;Index count=0;std::map<std::string,Index> stats;
    for(size_t part=0;part<continuations.size();++part) {
      const auto first=Index(part)*chunk;auto& q=continuations[part];Tensor partial;
      // Each physical slice retains all of its connected windows. Parameters
      // and optimizer are shared; no update occurs until every slice is done.
      for(Index window=0;window<c.windows;++window) {
        const auto cursor=position+window*p.tokens;
        auto positions=at::arange(cursor,cursor+p.tokens,at::kLong).reshape({1,-1});
        auto samples=at::arange(first,first+q.batch_size,at::kLong).reshape({-1,1});
        auto ids=(positions*7+samples*3).remainder(p.vocab).to(device);
        auto values=at::embedding(f.embedding,ids);std::vector<tide::External> external;
        for(Index b=0;b<q.batch_size;++b)for(Index t=0;t<p.tokens;++t)external.push_back({b,0,cursor+t,(cursor+t)*p.stride,values[b][t]});
        const auto stop=(cursor+p.tokens)*p.stride;
        auto result=greedy?greedy->run(q,external,stop,stop):streaming->run(q,external,stop,stop);
        q=result.continuation;
        if(c.family=="settle")result=f.settle->project(result);
        count+=result.outputs.size();auto value=objective(result,f,p,c.windows,first);
        if(value.defined())partial=partial.defined()?partial+value:value;
        for(const auto& [name,x]:result.stats)stats[name]=name.rfind("max_",0)==0?std::max(stats[name],x):stats[name]+x;
        if(diagnostics)window_json(*diagnostics,step,result,first,chunk<p.batch?p.batch:0);
      }
      if(partial.defined()){
        if(!at::isfinite(partial).all().item<bool>())throw std::runtime_error("nonfinite consumer loss");
        if(optimizer)optimizer->backward(partial);
        loss=loss.defined()?loss+partial.detach():partial.detach();
      }
      if(optimizer)detach(q);
    }
    position+=c.windows*p.tokens;
    double sample_seconds=-1;
    if(c.phase_timing&&optimizer){sync();sample_seconds=seconds(start);}
    if(optimizer) {
      std::map<std::string,std::vector<Tensor>> groups;
      for(const auto& owner:f.parameters.owners())if(owner.value.grad().defined())
        groups[owner.value.device().str()].push_back(at::isfinite(owner.value.grad()).all());
      std::vector<Tensor> flags;for(const auto& [_,values]:groups)flags.push_back(at::stack(values).all().to(device));
      if(!flags.empty()&&!at::stack(flags).all().item<bool>())throw std::runtime_error("nonfinite gradient; optimizer not applied");
      if(diagnostics)parameters_json(*diagnostics,step,f.parameters,true);
      optimizer->step();
    }
    sync();const auto elapsed=seconds(start);
    phases.add(elapsed,sample_seconds,step<c.warmup);
    if(diagnostics)parameters_json(*diagnostics,step,f.parameters,false);
    if(step>=c.warmup){durations.push_back(elapsed);losses.push_back(loss.defined()?loss.detach().cpu().item<double>():0.);outputs.push_back(count);statistics.push_back(stats);}
    else warmup_times.push_back(elapsed);
    if(step+1==c.warmup)memory.capture("warmup");
  }
  memory.capture("measured",false);
  const bool within=admission.observe(memory);
  const std::string error="consumer memory estimate underestimated observed peak; retain failed run and recalibrate";
  std::ostringstream out;out<<std::setprecision(17);
  out<<"{\"schema\":\"tide-online-consumer-v1\",\"state\":"<<quoted(within?"passed":"failed");
  if(!within)out<<",\"failure_phase\":\"post_run_memory_calibration\",\"error\":"<<quoted(error);
  out<<",\"workload_sha256\":"<<quoted(p.sha)
     <<",\"packet_identity\":\"declared; launcher must verify v2 text against hashed JSON\",\"implementation\":\"libtorch\",\"family\":"<<quoted(c.family)
     <<",\"training\":"<<(c.training?"true":"false")<<",\"optimizer\":"<<(c.training?quoted(c.optimizer):"null")
     <<",\"parameters\":"<<p.parameters()<<",\"construction_seconds\":"<<construction<<",\"seconds\":[";
  for(size_t i=0;i<durations.size();++i){if(i)out<<',';out<<durations[i];}
  out<<"],\"warmup_seconds\":[";for(size_t i=0;i<warmup_times.size();++i){if(i)out<<',';out<<warmup_times[i];}
  out<<"],\"losses\":[";for(size_t i=0;i<losses.size();++i){if(i)out<<',';if(outputs[i])out<<losses[i];else out<<"null";}
  out<<"],\"outputs\":[";for(size_t i=0;i<outputs.size();++i){if(i)out<<',';out<<outputs[i];}
  out<<"],\"statistics\":[";for(size_t i=0;i<statistics.size();++i){if(i)out<<',';out<<'{';bool first=true;
    for(const auto& [name,value]:statistics[i]){if(!first)out<<',';first=false;out<<quoted(name)<<':'<<value;}out<<'}';}
  out<<"],\"windows_per_step\":"<<c.windows<<",\"warmup_steps\":"<<c.warmup<<",\"measured_steps\":"<<c.steps
     <<",\"input_tokens_per_step\":"<<p.batch*p.tokens*c.windows<<",\"final_cut\":"<<continuations.front().cut
     <<",\"threads\":"<<c.threads<<",\"parameter_budget\":"<<c.parameter_budget<<",\"diagnostics\":"<<(c.diagnostics?"true":"false")
     <<",\"host_execution\":{\"workers\":"<<c.workers<<",\"packed_sources\":"<<(c.packed_sources?"true":"false")
     <<",\"batch_next\":"<<(c.batch_next?"true":"false")<<'}'
     <<",\"batch_execution\":{\"logical_batch\":"<<p.batch<<",\"requested_sample_chunk_rows\":"<<c.sample_chunk_rows
     <<",\"effective_sample_chunk_rows\":"<<chunk<<",\"physical_chunks\":"<<continuations.size()<<'}'
     <<",\"runtime\":{\"device\":"<<quoted(device.str())<<",\"dtype\":"<<quoted(portable_torch::dtype_name(c.runtime.dtype))
     <<",\"backend\":"<<quoted(portable_torch::compiled_backend())<<",\"resolution_reason\":"<<quoted(portable_torch::resolution_reason(c.runtime,device))
     <<",\"schedule\":"<<quoted(c.schedule)<<",\"preset\":"<<quoted(c.placement.preset)<<",\"placement\":{";
  bool first=true;for(const auto& [name,value]:placement.record()){if(!first)out<<',';first=false;out<<quoted(name)<<':'<<quoted(value);}out<<"}}"
     <<",\"payload_placement\":"<<eager_placement_json(c,owner_plan,devices)
     <<",\"memory\":"<<memory.json()<<",\"memory_admission\":"<<admission.json()<<",\"phase_timing\":"<<phases.json()
     <<",\"precision\":{\"payload\":"<<quoted(portable_torch::dtype_name(c.runtime.dtype))
     <<",\"loss\":"<<quoted(c.runtime.dtype==at::kHalf?"float32":portable_torch::dtype_name(c.runtime.dtype))
     <<",\"optimizer_masters\":"<<(c.training?quoted(c.runtime.dtype==at::kHalf?"float32":portable_torch::dtype_name(c.runtime.dtype)):"null")
     <<",\"gradient_accumulation\":\"payload\",\"loss_scale\":"<<c.loss_scale<<'}'
     <<",\"timing\":\"input preparation/upload + online forward + head/loss + backward + finite checks + detach/optimizer + synchronization; no reference\"}\n";
  if(!within)throw RecordedFailure(error,out.str());
  return out.str();
}
} // namespace tide_flow
