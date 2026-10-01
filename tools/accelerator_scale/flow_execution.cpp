#include "flow_execution.h"
#include <tide/frontier.h>
#include <tide/stream.h>
#include <stdexcept>

namespace accelerator_scale::flows {
HostWindow host_window(const pdg_scale::Config& c,Fixture& fixture,const Placement& placement,
                       const std::string& scheduler,Options options,const Tensor& ids,const std::vector<Tensor>& roots) {
  auto& f=fixture.values;HostWindow out;
  if(ids.scalar_type()!=at::kLong || ids.sizes()!=at::IntArrayRef({c.steps,c.batch})
      || (!roots.empty() && roots.size()!=size_t(c.steps)))throw std::invalid_argument("flow input shape mismatch");
  std::vector<Tensor> embeddings;std::vector<External> external;
  for(Index token=0;token<c.steps;++token) {
    auto x=f.embedding.index_select(0,ids[token].to(f.embedding.device()));
    if(!roots.empty())x=x+roots[token].to(x.device());
    embeddings.push_back(x);
    for(Index b=0;b<c.batch;++b)external.push_back({b,0,token,token*fixture.period,x[b]});
  }
  Continuation q;q.identity=f.graph.identity;q.batch_size=c.batch;
  const auto stop=c.steps*fixture.period;
  if(scheduler=="resident") {
    Resident cursor(f.graph,f.model,options,c.batch,placement);
    auto result=cursor.advance(external,stop,stop);
    out.result.outputs=std::move(result.outputs);out.result.trace=std::move(result.trace);
    out.result.messages=std::move(result.messages);out.result.stats=std::move(result.stats);
    if(options.trace)out.result.continuation=cursor.snapshot();
  } else if(scheduler=="streaming") {
    Streaming executor(f.graph,f.model,options);out.result=executor.run(q,external,stop,stop);
  } else if(scheduler=="frontier") {
    Frontier executor(f.graph,f.model,options);out.result=executor.run(q,external,stop,stop);
  } else if(scheduler=="settle") {
    if(!fixture.settle || !placement.devices.front().is_cpu())
      throw std::invalid_argument("direct native Settle flow currently requires its CPU body owners");
    SettleExecutor executor(*fixture.settle,fixture.body_model,options,"frontier");
    // Independent token roots stay independent. Stacking a whole window before
    // slicing the public Tensor input would connect unused token leaves to zero.
    for(const auto& embedding:embeddings) {
      auto piece=executor.run(q,embedding.unsqueeze(1));q=std::move(piece.continuation);
      out.result.trace.insert(out.result.trace.end(),piece.trace.begin(),piece.trace.end());
      out.result.messages.insert(out.result.messages.end(),piece.messages.begin(),piece.messages.end());
      out.result.outputs.insert(out.result.outputs.end(),piece.outputs.begin(),piece.outputs.end());
      for(const auto& [key,value]:piece.stats)out.result.stats[key]+=value;
    }
    out.result.continuation=std::move(q);
  } else throw std::invalid_argument("unknown complete host scheduler");
  // Sum every declared output at its sealed position, in canonical time order.
  // A missing output stays distinguishable in result.outputs and trace.
  std::vector<std::vector<Tensor>> hidden(c.steps);
  for(auto& rows:hidden)for(Index b=0;b<c.batch;++b)rows.push_back(at::zeros({c.width},f.head.options()));
  for(const auto& output:out.result.outputs) {
    const auto token=output.time/fixture.period;
    if(token<0 || token>=c.steps)throw std::logic_error("output outside declared complete window");
    auto& h=hidden[token][output.batch];h=h+output.value.to(f.head.device());
  }
  std::vector<Tensor> losses;DenseLinear head(c.head_workers);
  for(Index token=0;token<c.steps;++token) {
    auto logits=project(at::stack(hidden[token]),f.head,false,&head);out.logits.push_back(logits);
    if(c.grad) {
      auto target=(ids[token].to(logits.device())+1).remainder(c.vocab);
      losses.push_back(at::cross_entropy_loss(logits.to(at::kFloat),target));
    }
  }
  if(c.grad)out.loss=at::stack(losses).mean();return out;
}
}  // namespace accelerator_scale::flows
