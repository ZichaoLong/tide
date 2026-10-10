#include "tide/state_vjp.h"
#include "tide/operator_work.h"
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/saved_variable.h>
#include <sstream>
#include <stdexcept>

namespace tide {
namespace {
using torch::autograd::variable_list;
struct Request {
  State old;
  Tensor content;
  std::vector<SourceInput> sources;
  std::vector<SlotValue> contributions;
  Index time;
  ContentView view() const { return {content,sources,contributions}; }
};
template<class F> void tensors(Request& row,F f) {
  f(row.old.value);
  for (auto& [name,t]:row.old.slots) f(t);
  f(row.content);
  for (auto& s:row.sources) { f(s.atom.value); f(s.scale); }
  for (auto& s:row.contributions) f(s.value);
}
template<class F> void tensors(NodeWeights& w,F f) {
  f(w.decay); f(w.weight); f(w.bias); f(w.read);
  for (auto& [name,t]:w.extra) f(t);
}
variable_list values(const State& s) {
  variable_list result{s.value};
  for (const auto& [name,t]:s.slots) result.push_back(t);
  return result;
}
std::string signature(Request r) {
  // Only declared builtins enter here. Their branches use these coordinates,
  // shapes and source slots, never payload values or a CPU-computed trajectory.
  std::ostringstream s;
  s << r.time << ':' << r.old.last_time << ':' << r.old.observations;
  for (const auto& [name,t]:r.old.slots) s << ':' << name;
  s << ":sources" << r.sources.size();
  for (const auto& a:r.sources) s << ':' << a.slot;
  s << ":contributions" << r.contributions.size();
  for (const auto& a:r.contributions) s << ':' << a.slot;
  tensors(r,[&](Tensor& t) { s << '/' << t.sizes() << t.device() << int(t.scalar_type()) << t.requires_grad(); });
  return s.str();
}
// Each inner graph has identical row structure. Backpropagate one output-slot
// activity pattern at a time, then return only that pattern's participating
// rows. This removes stack's spurious cross-row zeros without inspecting any
// gradient value; parameter contributions still accumulate across every group.
class StateVjp final : public torch::autograd::Node {
 public:
  size_t weights=0,row_inputs=0,row_outputs=0,rows=0;
  std::vector<torch::autograd::SavedVariable> originals,leaves,outputs;
  std::vector<size_t> output_map;
  variable_list apply(variable_list&& bars) override {
    if (at::GradMode::is_enabled()) throw std::invalid_argument("batched State supports first-order VJP only");
    if (originals.empty()) throw std::runtime_error("State VJP graph already released; use retain_graph");
    for (auto& x:originals) x.unpack(); // Version checks on all captured inputs.
    variable_list xs,ys;
    for (auto& x:leaves) xs.push_back(x.unpack());
    for (auto& y:outputs) ys.push_back(y.unpack());
    variable_list full(rows*row_outputs),result(xs.size());
    for (size_t i=0;i<bars.size();++i) full[output_map[i]]=bars[i];
    std::map<std::vector<size_t>,std::vector<size_t>> groups;
    for (size_t r=0;r<rows;++r) {
      std::vector<size_t> mask;
      for (size_t j=0;j<row_outputs;++j) if (full[r*row_outputs+j].defined()) mask.push_back(j);
      if (!mask.empty()) groups[mask].push_back(r);
    }
    for (const auto& [mask,used]:groups) {
      variable_list roots,grads,inputs;
      std::vector<size_t> input_ids;
      for (auto r:used) for (auto j:mask) {
        roots.push_back(ys[r*row_outputs+j]); grads.push_back(full[r*row_outputs+j]);
      }
      auto append=[&](size_t i) { if (xs[i].requires_grad() && should_compute_output(i)) {
        input_ids.push_back(i); inputs.push_back(xs[i]);
      }};
      for (size_t i=0;i<weights;++i) append(i);
      for (auto r:used) for (size_t j=0;j<row_inputs;++j) append(weights+r*row_inputs+j);
      if (inputs.empty()) continue;
      const auto grads_in=torch::autograd::grad(roots,inputs,grads,true,false,true);
      for (size_t i=0;i<input_ids.size();++i) if (grads_in[i].defined()) {
        auto& out=result[input_ids[i]];
        out=out.defined()?out+grads_in[i]:grads_in[i];
      }
    }
    return result;
  }
  void release_variables() override {
    originals.clear(); leaves.clear(); outputs.clear();
  }
  std::string name() const override { return "tide::BatchedStateVjp"; }
};
std::vector<State> bind_group(const NodeWeights& weights,std::vector<Request> rows,const std::vector<State>& numeric) {
  auto node=std::make_shared<StateVjp>();
  auto w=weights;
  variable_list inputs,leaves;
  auto leaf=[&](Tensor& t) {
    inputs.push_back(t);
    t=t.detach().set_requires_grad(t.requires_grad());
    leaves.push_back(t);
  };
  tensors(w,leaf); node->weights=inputs.size();
  for (auto& row:rows) tensors(row,leaf);
  node->rows=rows.size();node->row_inputs=(inputs.size()-node->weights)/rows.size();
  node->set_next_edges(torch::autograd::collect_next_edges(inputs));
  for (const auto& t:inputs) node->originals.emplace_back(t,false);
  for (const auto& t:leaves) node->leaves.emplace_back(t,false);
  std::vector<State> old;
  variable_list content;
  ContentViews views;
  std::vector<Index> times;
  for (const auto& row:rows) { old.push_back(row.old); content.push_back(row.content); views.push_back(row.view()); times.push_back(row.time); }
  std::vector<State> semantic;
  {
    at::AutoGradMode enable(true);
    semantic=w.kernel->batch_graph(w,old,at::stack(content),times,views);
  }
  if (semantic.size()!=numeric.size()) throw std::invalid_argument("State VJP changed batch length");
  auto result=numeric;
  node->row_outputs=1+numeric[0].slots.size();
  for (size_t i=0;i<rows.size();++i) {
    const auto& s=semantic[i];const auto& n=numeric[i];
    if (s.last_time!=n.last_time || s.observations!=n.observations || s.slots.size()!=n.slots.size())
      throw std::invalid_argument("State VJP changed state metadata");
    for (const auto& [name,t]:n.slots) if (!s.slots.count(name)) throw std::invalid_argument("State VJP changed slots");
    auto refs=values(s),numeric_values=values(n);
    variable_list bound;
    for (size_t j=0;j<refs.size();++j) {
      if (refs[j].sizes()!=numeric_values[j].sizes() || refs[j].device()!=numeric_values[j].device()
          || refs[j].scalar_type()!=numeric_values[j].scalar_type()) throw std::invalid_argument("State VJP changed tensor metadata");
      node->outputs.emplace_back(refs[j],false);
      auto value=numeric_values[j].detach().clone();
      if (refs[j].requires_grad()) {
        node->output_map.push_back(i*refs.size()+j);
        torch::autograd::set_history(value,node);
      }
      bound.push_back(value);
    }
    result[i].value=bound[0];size_t j=1;
    for (auto& [name,t]:result[i].slots) t=bound[j++];
  }
  return result;
}
}
std::vector<State> state_batch_vjp(const NodeWeights& w,const std::vector<State>& old,
    const ContentViews& views,const std::vector<Index>& times,const std::vector<State>& numeric) {
  work::StateReplayTimer timer(work::StateBatchGraphNs);
  std::map<std::string,std::vector<size_t>> groups;
  std::vector<Request> requests;
  for (size_t i=0;i<old.size();++i) {
    requests.push_back({old[i],views[i].value,{views[i].sources.begin(),views[i].sources.end()},
                       {views[i].contributions.begin(),views[i].contributions.end()},times[i]});
    groups[signature(requests.back())].push_back(i);
  }
  auto result=numeric;
  for (const auto& [key,ids]:groups) {
    std::vector<Request> rows;std::vector<State> packed;
    for (auto i:ids) { rows.push_back(requests[i]); packed.push_back(numeric[i]); }
    auto bound=bind_group(w,std::move(rows),packed);
    for (size_t i=0;i<ids.size();++i) result[ids[i]]=std::move(bound[i]);
  }
  return result;
}
void state_sequence_vjp(const NodeWeights& w,const std::vector<State>& old,
    const PackedSequence& p,std::vector<State>& numeric,std::vector<State>& previous) {
  auto current=old;previous.resize(numeric.size());
  for (Index depth=0;;++depth) {
    std::vector<size_t> owners,ids;
    std::vector<State> initial,packed;
    ContentViews views;std::vector<Index> times;
    for (size_t i=0;i<old.size();++i) if (p.offsets[i]+depth<p.offsets[i+1]) {
      const auto j=p.offsets[i]+depth; owners.push_back(i);ids.push_back(j);
      previous[j]=current[i];initial.push_back(current[i]);packed.push_back(numeric[j]);
      views.push_back(p.views[j]);times.push_back(p.times[j]);
    }
    if (ids.empty()) break;
    auto bound=state_batch_vjp(w,initial,views,times,packed);
    for (size_t i=0;i<ids.size();++i) current[owners[i]]=numeric[ids[i]]=std::move(bound[i]);
  }
}
}
