#include "bounded_check.h"
#include "graph_replay.h"
#include "../../cpp/bench/streaming.h"
#include <iostream>

namespace accelerator_scale::bounded {
void check_replay(pdg_scale::Config c,const pdg_scale::Topology& topology,at::Device device,Index devices) {
  if(c.width>64 || c.batch>8 || c.steps>6)throw std::invalid_argument("bounded replay parity requires small tensors");
  at::NoGradGuard guard;std::vector<at::Device> targets;
  for(Index i=0;i<devices;++i)targets.emplace_back(device.type(),device.index()+i);
  GraphReplay replay(targets);auto actual=fixture(c,topology,true);
  auto placement=place(actual,device,devices,"locality",true);placement.scoring={"model","model",at::kFloat};
  Program p(actual,topology,placement,Limits{c.steps,c.batch,int64_t(4)<<30,true,true});
  auto initial=(at::arange(c.steps).unsqueeze(1)*7+at::arange(c.batch).unsqueeze(0)*3).remainder(c.vocab).to(at::kLong);
  auto ids=initial.to(device);Window window;
  {auto warmup=p.run(ids);replay.synchronize();auto ref=fixture(c,topology,false);auto expected=oracle(ref,c,topology,initial);
   try{tide_bench::compare(export_result(p,warmup),expected.result,true,c.runtime.dtype,std::nullopt,c.check_rtol,c.check_atol);}
   catch(const std::exception& e){throw std::runtime_error(std::string("peer warmup: ")+e.what());}
   std::cout<<"PASS bounded eager peer warmup memory="<<c.memory<<'\n'<<std::flush;}
  replay.capture([&]{window=p.run(ids);});
  for(Index trial=0;trial<3;++trial) {
    auto input=(initial+trial*5).remainder(c.vocab);ids.copy_(input);replay.synchronize();
    replay.replay();replay.synchronize();
    auto ref=fixture(c,topology,false);auto expected=oracle(ref,c,topology,input);
    const double rtol=c.check_rtol,atol=c.check_atol;
    auto observed=export_result(p,window);
    // Diagnose the earliest public event before checking the final snapshot;
    // a late state mismatch otherwise hides the origin of replay corruption.
    if(observed.trace.size()==expected.result.trace.size())for(size_t i=0;i<observed.trace.size();++i) {
      const auto& a=observed.trace[i];const auto& b=expected.result.trace[i];
      if(std::tie(a.time,a.batch,a.node)!=std::tie(b.time,b.batch,b.node))break;
      const auto label="replay trial="+std::to_string(trial)+" event="+std::to_string(a.time)+":"+std::to_string(a.batch)+":"+std::to_string(a.node);
      if(a.fiber.size()==b.fiber.size())for(size_t j=0;j<a.fiber.size();++j)
        close(a.fiber[j].value,b.fiber[j].value,rtol,atol,label+" fiber="+std::to_string(a.fiber[j].source));
      if(a.contributions.size()==b.contributions.size())for(size_t j=0;j<a.contributions.size();++j)
        close(a.contributions[j].value,b.contributions[j].value,rtol,atol,label+" contribution="+std::to_string(a.contributions[j].slot));
      close(a.content,b.content,rtol,atol,label+" content");
      close(a.proposal,b.proposal,rtol,atol,label+" proposal");
      close(a.descriptor,b.descriptor,rtol,atol,label+" descriptor");
      close(a.control,b.control,rtol,atol,label+" control");
      if(a.active==b.active)close(a.full,b.full,rtol,atol,label+" Full");
      if(a.emitted.size()==b.emitted.size())for(size_t j=0;j<a.emitted.size();++j)
        close(a.emitted[j].value,b.emitted[j].value,rtol,atol,label+" emitted="+std::to_string(a.emitted[j].slot));
    }
    tide_bench::compare(observed,expected.result,true,c.runtime.dtype,std::nullopt,rtol,atol);
    for(Index token=0;token<c.steps;++token)close(window.logits[token].data,expected.logits[token],rtol,atol,"replay logits");
    std::cout<<"PASS device-resident bounded replay trial="<<trial<<" changed_inputs=true memory="<<c.memory<<'\n'<<std::flush;
  }
}
}  // namespace accelerator_scale::bounded
