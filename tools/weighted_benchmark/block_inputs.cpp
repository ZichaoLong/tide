#include "block_inputs.h"
#include <ATen/core/grad_mode.h>
#include <algorithm>
#include <limits>
namespace weighted_benchmark {
BlockInputs block_inputs(const SettleGraph& spec,const Fixture& f){
  at::NoGradGuard guard;Continuation q;q.identity=f.graph.identity;q.batch_size=f.samples;
  spec.validate_alignment(q,f.inputs);
  const auto width=std::get<weighted::KVConfig>(weighted::node_binding(f.model.nodes.at(0)).memory).width+1;
  const auto ports=Index(f.graph.inputs.size());
  BlockInputs out{at::arange(f.positions,at::kLong).unsqueeze(0).expand({f.samples,f.positions}).contiguous(),
    at::zeros({f.samples,f.positions,ports,width},at::kFloat),at::zeros({f.samples,f.positions,ports},at::kBool)};
  std::map<std::tuple<Index,Index,Index>,const External*> ordered;
  for(const auto& x:f.inputs){
    if(x.batch<0||x.batch>=f.samples||x.port<0||x.port>=ports)throw std::invalid_argument("block fixture input domain differs");
    const auto position=(x.time-spec.rank(f.graph.inputs[x.port]))/spec.stride();
    if(position<0||position>=f.positions||!ordered.emplace(std::make_tuple(x.batch,x.port,position),&x).second)
      throw std::invalid_argument("block fixture has duplicate/outside input positions");
  }
  std::map<std::pair<Index,Index>,Index> occurrences;
  for(const auto& [key,entry]:ordered){const auto [b,p,t]=key;const auto& x=*entry;
    if(x.position!=occurrences[{b,p}]++||!x.value.device().is_cpu()||x.value.scalar_type()!=at::kFloat||x.value.sizes()!=at::IntArrayRef{width})
      throw std::invalid_argument("block fixture input occurrence/payload differs");
    out.payload[b][t][p].copy_(x.value);out.present[b][t][p]=true;
  }
  return out;
}
OutputObjective block_objective(const Tensor& c,const Tensor& g,const Tensor& present,double scale,bool roots){
  OutputObjective out;const auto mask=present.unsqueeze(-1),numeric=c.unsqueeze(-1)*g;
  out.value=at::where(mask,numeric.square(),at::zeros_like(numeric)).sum()*scale;
  if(roots){
    out.strength_root=at::where(present,2*scale*c*g.square().sum(-1),at::zeros_like(c));
    out.content_root=at::where(mask,2*scale*c.square().unsqueeze(-1)*g,at::zeros_like(g));
  }
  return out;
}
std::vector<Output> block_outputs(const SettleGraph& spec,Index length,const Tensor& c,const Tensor& g,
                                 const Tensor& present,const Tensor& times){
  at::NoGradGuard guard;const auto payload=at::cat({c.unsqueeze(-1),g},-1).detach().cpu();
  const auto flags=present.cpu(),clock=times.cpu();const auto indices=at::nonzero(flags);
  std::vector<Output> out;for(Index j=0;j<indices.size(0);++j){
    const auto frame=indices[j][0].item<Index>(),port=indices[j][1].item<Index>();
    out.push_back({frame/length,clock[frame][port].item<Index>(),port,payload[frame][port].clone()});
  }
  std::sort(out.begin(),out.end(),[&](const auto& a,const auto& b){return
    std::make_tuple(a.time,a.batch,spec.graph().outputs[a.port],a.port)<
    std::make_tuple(b.time,b.batch,spec.graph().outputs[b.port],b.port);});return out;
}
}
