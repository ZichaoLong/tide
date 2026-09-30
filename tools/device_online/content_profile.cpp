#include "content_profile.h"
#include "tide/kernel.h"
#include "tide/ops.h"
#include <ATen/core/grad_mode.h>
#include <stdexcept>

namespace tide::device_online {
ContentProfile::ContentProfile(Graph g,Model m,at::Device device):graph(std::move(g)),model(std::move(m)) {
  if(at::GradMode::is_enabled()||device.type()!=c10::DeviceType::PrivateUse1)
    throw std::invalid_argument("content flow requires explicit no-grad NPU");
  graph.compile();
  if(!graph.origins.empty())throw std::invalid_argument("content flow input-origin projections are not implemented");
  for(const auto& n:graph.nodes) {
    if((!n.identity&&n.memory!="identity"&&n.memory!="ema")||(!n.identity&&n.full!="identity"&&n.full!="tanh")
        ||n.aggregation!="sum"||n.readout!="linear-v1"||n.next_state!="adopt-v1"
        ||n.emission!="broadcast"||n.state_clock!=StateClock{})
      throw std::invalid_argument("content flow module contract unavailable");
    owners.push_back(n.region);
  }
  for(const auto& r:graph.regions) {
    if((r.read_mode!="content"&&r.read_mode!="old"&&r.read_mode!="proposal")||(r.selector!="count-v1"&&r.selector!="positive-v1"))
      throw std::invalid_argument("content flow requires a built-in linear Read mode and count/positive selection");
    policies.push_back({r.budget,r.count_priority,r.selector=="positive-v1"});
    causal_regions.push_back(r.read_mode!="content"&&!r.observe_all);
    all_content&=r.read_mode=="content";
  }
  for(const auto& n:graph.nodes)if(n.clear&&graph.regions[n.region].read_mode!="content")causal_regions[n.region]=1;
  for(const auto& w:model.nodes) {
    if(w.kernel||w.read_kernel||w.next_kernel||w.aggregate_kernel||w.full_kernel)
      throw std::invalid_argument("content flow requires built-in module declarations, not custom kernel handles");
    if(!w.bias.defined()||w.bias.device()!=at::Device(at::kCPU)||w.bias.scalar_type()!=at::kFloat)
      throw std::invalid_argument("content flow currently requires CPU FP32 parameter inputs");
  }
  for(const auto& w:model.regions)if(w.kernel)throw std::invalid_argument("custom region kernel unavailable");
  configure_model(graph,model);validate_model(graph,model);width=model.width();
  // Take independent values, preserving no user-owned mutable parameter alias.
  auto copy=[](const Tensor& x){return x.detach().clone();};
  for(auto& w:model.nodes){w.decay=copy(w.decay);w.weight=copy(w.weight);w.bias=copy(w.bias);w.read=copy(w.read);}
  for(auto group:{&model.input_scale,&model.agg_scale,&model.edge_scale,&model.output_scale})for(auto& x:*group)x=copy(x);
  std::vector<int64_t> metadata,settings,modes;
  std::vector<Tensor> weights,reads,decays;
  for(size_t p=0;p<graph.inputs.size();++p) {
    metadata.insert(metadata.end(),{graph.inputs[p],graph.source_domain->input[p]});weights.push_back(model.input_scale[p]);
  }
  for(size_t e=0;e<graph.edges.size();++e) {
    const auto& edge=graph.edges[e];wires.push_back({edge.source,edge.target,edge.delay});
    metadata.insert(metadata.end(),{edge.target,graph.source_domain->edge_target[e]});weights.push_back(model.agg_scale[e]);
  }
  // Physical nonempty tensors are required even for a graph with no sources.
  if(weights.empty()){metadata={0,0};weights.push_back(at::zeros({},at::kFloat));}
  for(size_t n=0;n<graph.nodes.size();++n) {
    const auto& node=graph.nodes[n];
    settings.insert(settings.end(),{!node.identity&&node.memory=="ema",node.clear,graph.regions[node.region].observe_all});
    reads.push_back(node.identity?at::zeros_like(model.nodes[n].read):model.nodes[n].read);
    const auto& mode=graph.regions[node.region].read_mode;
    modes.push_back(node.identity?-1:mode=="content"?0:mode=="old"?1:2);
    decays.push_back(model.nodes[n].decay);
  }
  sources=at::tensor(metadata,at::kLong).reshape({-1,2}).to(device);scales=at::stack(weights).to(device);
  read=at::stack(reads).to(device);decay=at::stack(decays).to(device);
  read_modes=at::tensor(modes,at::kLong).to(device);
  config=at::tensor(settings,at::kLong).reshape({-1,3}).to(device);
  auto pack_scales=[&](const std::vector<Tensor>& x){return (x.empty()?at::zeros({1},at::kFloat):at::stack(x)).reshape({-1,1}).to(device);};
  edge_scales=pack_scales(model.edge_scale);output_scales=pack_scales(model.output_scale);
  output_nodes=at::tensor(graph.outputs.empty()?std::vector<int64_t>{0}:graph.outputs,at::kLong).to(device);
}
} // namespace tide::device_online
