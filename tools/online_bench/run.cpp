#include "consumer.h"
#include "memory.h"
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
Tensor objective(const tide::Result& r,const Fixture& f,const Packet& p,Index windows) {
  if(r.outputs.empty())return {};
  std::vector<Tensor> rows;std::vector<Index> labels;
  for(const auto& x:r.outputs){rows.push_back(x.value);labels.push_back(((x.time/p.stride+1)*7+x.batch*3)%p.vocab);}
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
  auto start=Clock::now();MemoryRecord memory({device});auto f=fixture(p,c,device);
  auto placement=tide::resolve_placement(c.placement,device);auto placed=tide::place_model(f.graph,f.model,c.placement);
  tide::Options options;options.packed=true;options.prefill=c.schedule=="prefill";options.trace=c.diagnostics;
  options.full_autograd="batched";options.aggregate_autograd="batched";
  std::unique_ptr<tide::Streaming> streaming;std::unique_ptr<tide::Greedy> greedy;
  if(c.schedule=="prefill")greedy=std::make_unique<tide::Greedy>(f.graph,placed,options);
  else streaming=std::make_unique<tide::Streaming>(f.graph,placed,options);
  std::unique_ptr<tide::NamedOptimizer> optimizer;
  tide::OptimizerGroup group;group.lr=.0001;group.weight_decay=.001;group.momentum=.25;group.eps=1e-6;
  for(const auto& owner:f.parameters.owners())group.parameters.push_back(owner.canonical);
  if(c.training) {
    if(c.optimizer=="sgd")optimizer=std::make_unique<tide::SGD>(f.parameters,std::vector<tide::OptimizerGroup>{group});
    else optimizer=std::make_unique<tide::AdamW>(f.parameters,std::vector<tide::OptimizerGroup>{group});
  }
  tide::Continuation q;q.identity=f.graph.identity;q.batch_size=p.batch;
  portable_torch::synchronize(device);const auto construction=seconds(start);
  memory.capture("construction");
  std::vector<double> durations,losses,warmup_times;std::vector<Index> outputs;std::vector<std::map<std::string,Index>> statistics;
  Index position=0;at::AutoGradMode mode(c.training);
  for(Index step=0;step<c.steps+c.warmup;++step) {
    portable_torch::synchronize(device);start=Clock::now();if(optimizer)optimizer->zero_grad();
    Tensor loss;Index count=0;std::map<std::string,Index> stats;
    for(Index window=0;window<c.windows;++window) {
      auto positions=at::arange(position,position+p.tokens,at::kLong).reshape({1,-1});
      auto samples=at::arange(p.batch,at::kLong).reshape({-1,1});
      auto ids=(positions*7+samples*3).remainder(p.vocab).to(device);
      auto values=at::embedding(f.embedding,ids);std::vector<tide::External> external;
      for(Index b=0;b<p.batch;++b)for(Index t=0;t<p.tokens;++t)external.push_back({b,0,position+t,(position+t)*p.stride,values[b][t]});
      const auto stop=(position+p.tokens)*p.stride;
      auto result=greedy?greedy->run(q,external,stop,stop):streaming->run(q,external,stop,stop);
      q=result.continuation;position+=p.tokens;
      if(c.family=="settle")result=f.settle->project(result);
      count+=result.outputs.size();auto value=objective(result,f,p,c.windows);
      if(value.defined())loss=loss.defined()?loss+value:value;
      for(const auto& [name,x]:result.stats)stats[name]=name.rfind("max_",0)==0?std::max(stats[name],x):stats[name]+x;
      if(diagnostics)window_json(*diagnostics,step,result);
    }
    if(loss.defined()){
      if(!at::isfinite(loss).all().item<bool>())throw std::runtime_error("nonfinite consumer loss");
      if(optimizer)loss.backward();
    }
    if(optimizer) {
      std::vector<Tensor> flags;for(const auto& owner:f.parameters.owners())if(owner.value.grad().defined())flags.push_back(at::isfinite(owner.value.grad()).all());
      if(!flags.empty()&&!at::stack(flags).all().item<bool>())throw std::runtime_error("nonfinite gradient; optimizer not applied");
      if(diagnostics)parameters_json(*diagnostics,step,f.parameters,true);
      detach(q);optimizer->step();
    }
    portable_torch::synchronize(device);const auto elapsed=seconds(start);
    if(diagnostics)parameters_json(*diagnostics,step,f.parameters,false);
    if(step>=c.warmup){durations.push_back(elapsed);losses.push_back(loss.defined()?loss.detach().cpu().item<double>():0.);outputs.push_back(count);statistics.push_back(stats);}
    else warmup_times.push_back(elapsed);
    if(step+1==c.warmup)memory.capture("warmup");
  }
  memory.capture("measured",false);
  std::ostringstream out;out<<std::setprecision(17);
  out<<"{\"schema\":\"tide-online-consumer-v1\",\"state\":\"passed\",\"workload_sha256\":"<<quoted(p.sha)
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
     <<",\"input_tokens_per_step\":"<<p.batch*p.tokens*c.windows<<",\"final_cut\":"<<q.cut
     <<",\"threads\":"<<c.threads<<",\"parameter_budget\":"<<c.parameter_budget<<",\"diagnostics\":"<<(c.diagnostics?"true":"false")
     <<",\"runtime\":{\"device\":"<<quoted(device.str())<<",\"dtype\":"<<quoted(portable_torch::dtype_name(c.runtime.dtype))
     <<",\"backend\":"<<quoted(portable_torch::compiled_backend())<<",\"resolution_reason\":"<<quoted(portable_torch::resolution_reason(c.runtime,device))
     <<",\"schedule\":"<<quoted(c.schedule)<<",\"preset\":"<<quoted(c.placement.preset)<<",\"placement\":{";
  bool first=true;for(const auto& [name,value]:placement.record()){if(!first)out<<',';first=false;out<<quoted(name)<<':'<<quoted(value);}out<<"}}"
     <<",\"memory\":"<<memory.json()
     <<",\"timing\":\"input preparation/upload + online forward + head/loss + backward + finite checks + detach/optimizer + synchronization; no reference\"}\n";
  return out.str();
}
} // namespace tide_flow
