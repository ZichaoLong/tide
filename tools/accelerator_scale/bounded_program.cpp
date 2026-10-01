#include "bounded.h"
#include "peer_transport.h"
#include <tide/kernel.h>
#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>

namespace accelerator_scale::bounded {
int64_t estimate_workspace(const pdg_scale::Topology& topology,const std::string& memory,
                          Index width,at::ScalarType dtype,const Limits& limits,bool training) {
  if(memory!="add" && memory!="attention")throw std::invalid_argument("unsupported bounded memory");
  if(width<1 || limits.tokens<1 || limits.tokens>32 || limits.batch<1 || limits.batch>512
      || topology.nodes<2 || topology.nodes>200000 || topology.layers<1 || topology.layers>8
      || topology.edges.size()>4000000)
    throw std::invalid_argument("invalid bounded capacity dimensions");
  std::set<Index> projected_nodes;for(const auto& edge:topology.edges)projected_nodes.insert(edge.source);
  const Index nodes=2*topology.nodes+1,edges=(Index(topology.edges.size())+1)*topology.layers;
  const Index owners=(memory=="attention"?3:1)*nodes+(nodes-1)+projected_nodes.size()+2;
  const long double b=limits.batch,d=width,t=limits.tokens*(topology.layers+1),s=c10::elementSize(dtype);
  const long double dep=limits.connectivity?owners+limits.tokens:1;
  long double bound=16*b*(nodes+edges)*(d*s+dep)*(limits.trace || training?t:2);
  if(memory=="attention")bound+=8*b*edges*t*d*s;
  if(bound>std::numeric_limits<int64_t>::max())throw std::overflow_error("bounded workspace accounting overflow");
  return static_cast<int64_t>(bound);
}
namespace {
Schedule historical_schedule(const pdg_scale::Fixture& f,const pdg_scale::Topology& topology,const Limits& limits) {
  Schedule plan{topology.layers+1,{},estimate_workspace(topology,
    f.graph.nodes[0].memory=="lh-add-repeat-v1"?"add":"attention",
    f.model.width(),f.embedding.scalar_type(),limits,at::GradMode::is_enabled())};
  std::vector<Index> rows(f.graph.nodes.size());
  for(const auto& edge:topology.edges) {
    const auto row=rows[edge.source]++;
    for(Index phase=0;phase<topology.layers;++phase)plan.edge_rows.push_back(row);
  }
  for(Index phase=0;phase<topology.layers;++phase)plan.edge_rows.push_back(-1);
  return plan;
}
}
Program::Program(pdg_scale::Fixture& f,const pdg_scale::Topology& topology,Placement placement,Limits limits)
    :Program(f,historical_schedule(f,topology,limits),std::move(placement),limits){}
Program::Program(pdg_scale::Fixture& f,Schedule schedule,Placement placement,Limits limits)
    : f_(f), period_(schedule.period), placement_(placement), limits_(limits),
      workspace_bound_(schedule.workspace_bound), leaves_(f.owners), edge_rows_(std::move(schedule.edge_rows)) {
  if (!placement.resident || limits.tokens < 1 || limits.tokens > 32 || limits.batch < 1
      || limits.batch > 512 || limits.max_workspace_bytes <= 0 || period_<1
      || period_>std::numeric_limits<Index>::max()/limits.tokens)
    throw std::invalid_argument("bounded scheduler requires resident transport and explicit positive finite capacities");
  f_.graph.compile(); configure_model(f_.graph, f_.model);
  if(f_.graph.inputs.size()!=1 || f_.graph.outputs.size()!=1 || edge_rows_.size()!=f_.graph.edges.size())
    throw std::invalid_argument("bounded schedule requires one encoded input/output and exact physical edge rows");
  for (const auto& n : f_.graph.nodes)
    if ((!n.identity && n.memory != "lh-add-repeat-v1" && n.memory != "lh-fiber-attention-all-softmax-repeat-v1")
        || n.next_state != "adopt-v1")
      throw std::invalid_argument("bounded scheduler supports Add/Attention and identity boundaries only");
  for (const auto& r : f_.graph.regions)
    if (!r.observe_all || r.read_mode != "proposal"
        || (r.selector != "lh-count-affect-v1" && r.selector != "count-v1"))
      throw std::invalid_argument("unsupported bounded selector/adoption policy");
  if (placement.scoring.dtype != at::kFloat || placement.scoring.read_device != "model"
      || placement.scoring.control_device != "model")
    throw std::invalid_argument("bounded scheduling requires explicit model FP32 Read and controls");
  for (size_t i=0; i<leaves_.size(); ++i) {
    if (!owner_.emplace(leaves_[i].storage().unsafeGetStorageImpl(), i).second)
      throw std::invalid_argument("bounded owner storage alias is unsupported");
  }
  for (auto d : placement.devices)
    leaf_ids_.emplace(d.str(), at::arange(leaves_.size()+limits.tokens,
      at::TensorOptions().dtype(at::kLong).device(d)));
  members_.resize(f_.graph.regions.size());
  for (size_t n=0; n<f_.graph.nodes.size(); ++n) members_[f_.graph.nodes[n].region].push_back(n);
  // Conservative tensor arena bound, including dense predication, cache growth,
  // training activations and connectivity. It is a refusal bound, not an OOM
  // guarantee; the launcher also enforces measured RSS/time/device limits.
  if (workspace_bound_<0 || workspace_bound_>limits.max_workspace_bytes)
    throw std::invalid_argument("bounded workspace estimate exceeds explicit capacity: "+std::to_string(workspace_bound_));
}
Tensor Program::empty_dependencies(at::Device d) const {
  return at::zeros({limits_.batch, limits_.connectivity ? Index(leaves_.size())+limits_.tokens : 1},
    at::TensorOptions().dtype(at::kBool).device(d));
}
Tensor Program::dependency(const Tensor& t, at::Device d) const {
  if (!limits_.connectivity) return empty_dependencies(d);
  const auto it=owner_.find(t.storage().unsafeGetStorageImpl());
  if (it==owner_.end()) return empty_dependencies(d);
  return (leaf_ids_.at(d.str())==it->second).unsqueeze(0).expand({limits_.batch,-1});
}
Tensor move(const Tensor& value,at::Device d) {
  // Materialize views before the SDK cross-device copy: strided/offset sources
  // otherwise enter its legacy Slice compiler path on the standalone SDK.
  return replay_peer_copy(value,d);
}
Value Program::copy(const Value& v, at::Device d) const { return {move(v.data,d),move(v.dependencies,d)}; }
State Program::initial(Index node) const {
  auto opts=f_.model.nodes[node].bias.options(); auto dep=empty_dependencies(opts.device());
  State s; s.value={at::zeros({limits_.batch,f_.model.width()},opts),dep};
  s.last=at::full({limits_.batch},-1,opts.dtype(at::kLong));
  s.observations=at::zeros_like(s.last);s.seen=at::zeros({limits_.batch},opts.dtype(at::kBool));
  if (!f_.graph.nodes[node].identity && f_.graph.nodes[node].memory!="lh-add-repeat-v1") {
    s.key=at::zeros({limits_.batch,0,4,f_.model.width()/4},opts);s.cache_value=at::zeros_like(s.key);
    s.bias=at::zeros({limits_.batch,0},opts);s.valid=at::zeros({limits_.batch,0},opts.dtype(at::kBool));
    s.cache_dependencies=dep;
  }
  return s;
}
Window Program::run(const Tensor& ids, const std::vector<Tensor>& external_roots) const {
  if (ids.scalar_type()!=at::kLong || ids.sizes()!=at::IntArrayRef({limits_.tokens,limits_.batch})
      || (!external_roots.empty() && external_roots.size()!=size_t(limits_.tokens)))
    throw std::invalid_argument("bounded fixed input shape/dtype mismatch");
  Window out;
  for (size_t n=0;n<f_.graph.nodes.size();++n)out.states.push_back(initial(n));
  for (const auto& members:members_) {
    auto opts=f_.model.nodes[members.front()].bias.options();
    auto counts=at::zeros({limits_.batch,Index(members.size())},opts.dtype(at::kLong));
    out.histories.push_back({counts,at::zeros_like(counts),
      at::full({limits_.batch},-1,opts.dtype(at::kLong)),at::zeros({limits_.batch},opts.dtype(at::kBool))});
  }
  std::map<Index,std::map<Index,std::vector<Message>>> queue;
  const Index period=period_, stop=limits_.tokens*period, width=f_.model.width();
  for (Index token=0;token<limits_.tokens;++token) {
    const auto d=f_.embedding.device();
    auto x=f_.embedding.index_select(0,move(ids[token],d));
    auto deps=dependency(f_.embedding,d);
    if (!external_roots.empty()) {
      x=x+move(external_roots[token],d);
      if(limits_.connectivity)deps=deps | (leaf_ids_.at(d.str())==Index(leaves_.size())+token).unsqueeze(0);
    }
    auto yes=at::ones({limits_.batch},x.options().dtype(at::kBool));
    const auto input_node=f_.graph.inputs[0];
    queue[token*period][input_node].push_back({input_node,token*period,0,0,token,f_.graph.source_domain->input[0],{x,deps},yes});
  }
  std::vector<Value> hidden(limits_.tokens);
  for (Index time=0;time<stop;++time) {
    auto it=queue.find(time);if(it==queue.end())continue; // static topology only
    auto arrived=std::move(it->second);queue.erase(it);
    std::map<Index,Event> events;
    for (auto& [node,fiber]:arrived) events.emplace(node,update(node,time,out.states[node],std::move(fiber)));
    for (Index r=0;r<Index(members_.size());++r) select(r,time,events,out.histories[r]);
    for (auto& [node,event]:events) {
      finish(event);out.states[node]=event.next;
      out.candidates.push_back(event.candidate);out.selected.push_back(event.active);
      const auto first=f_.graph.outgoing_ports.offsets[node];
      for(size_t j=0;j<event.emitted.size();++j) {
        const auto binding=f_.graph.outgoing_ports.bindings[first+event.emit_slots[j]];
        if(binding.kind==0) {
          auto v=event.emitted[j];v.data=masked_rows(v.data,event.active);
          v.dependencies=v.dependencies & event.active.unsqueeze(1);
          auto& h=hidden[time/period];
          if(h.data.defined()){h.data=h.data+v.data;h.dependencies=h.dependencies|v.dependencies;}
          else h=v;
          out.output_present.push_back(event.active);
        } else {
          const auto edge=binding.id;const auto& wire=f_.graph.edges[edge];
          Message m{wire.target,time+wire.delay,1,edge,time,f_.graph.source_domain->edge_target[edge],event.emitted[j],event.active};
          queue[m.time][m.node].push_back(m);if(limits_.trace)out.messages.push_back(m);
          out.edge_presence.push_back(event.active);
        }
      }
      if(limits_.trace)out.events.push_back(std::move(event));
    }
  }
  for(auto& [time,nodes]:queue)for(auto& [node,messages]:nodes)
    out.pending.insert(out.pending.end(),messages.begin(),messages.end());
  for(Index token=0;token<limits_.tokens;++token) {
    const auto d=f_.head.device();Value h;
    if(hidden[token].data.defined())h=copy(hidden[token],d);
    else h={at::zeros({limits_.batch,width},f_.head.options()),empty_dependencies(d)};
    out.logits.push_back({at::linear(h.data,f_.head),h.dependencies|dependency(f_.head,d)});
  }
  return out;
}
}  // namespace accelerator_scale::bounded
