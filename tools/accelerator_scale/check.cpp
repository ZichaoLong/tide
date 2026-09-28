#include "execution.h"
#include "check_vjp.h"
#include "../../cpp/bench/streaming.h"
#include <iostream>
#include <set>

namespace accelerator_scale {
namespace {
struct Run { Result result; std::vector<Tensor> leaves, roots; };
Run execute(pdg_scale::Config c, const pdg_scale::Topology& topology, at::Device device,
            Index devices, bool candidate, const std::string& policy, bool resident) {
  portable_torch::seed_runtime(at::Device(at::kCPU), c.runtime.seed);
  c.emission = candidate ? "row" : "slot";
  auto f = pdg_scale::fixture(c,topology);
  if (!candidate && at::GradMode::is_enabled()) {
    // Historical slot fixtures were forward-only: their emit views were made
    // under NoGradGuard. Rebind the same storage views with autograd enabled so
    // this independent scalar oracle differentiates the declared owner leaves.
    std::map<const c10::StorageImpl*,Tensor> owners;
    for (const auto& p:f.owners) owners[p.storage().unsafeGetStorageImpl()]=p;
    for (auto& w:f.model.nodes) for(auto& [name,p]:w.extra) {
      const auto it=owners.find(p.storage().unsafeGetStorageImpl());
      if(it!=owners.end() && !p.is_same(it->second))
        p=it->second.as_strided(p.sizes(),p.strides(),p.storage_offset());
    }
  }
  Placement placement;
  f.graph.compile(); const auto identity = f.graph.identity;
  if (candidate) placement = place(f,device,devices,policy,resident);
  f.graph.compile(); if (f.graph.identity != identity) throw std::runtime_error("placement changed graph identity");
  Options options; options.packed = candidate; options.trace = true;
  options.workers = candidate ? c.workers : 1;
  options.full_autograd = candidate ? c.full_autograd : "replay";
  options.aggregate_autograd = candidate ? c.aggregate_autograd : "replay";
  options.parallel_regions = candidate && c.parallel_regions;
  options.packed_sources = candidate && c.packed_sources;
  options.batch_next = candidate && c.batch_next;
  Execution cursor(f.graph,f.model,options,c.batch,placement);
  Run run; run.leaves = f.owners;
  std::vector<Tensor> logits;
  for (Index token = 0; token < c.steps; ++token) {
    auto ids = at::remainder(at::arange(c.batch,at::TensorOptions().dtype(at::kLong))*3+token*7,c.vocab);
    auto inputs = embed(f.embedding,ids,!(candidate && resident));
    auto root = at::zeros_like(inputs).set_requires_grad(true); run.leaves.push_back(root);
    inputs = inputs+root;
    std::vector<External> external;
    for (Index b = 0; b < c.batch; ++b) external.push_back({b,0,token,token*(topology.layers+1),inputs[b]});
    auto result = cursor.advance(external,(token+1)*(topology.layers+1),(token+1)*(topology.layers+1));
    run.result.trace.insert(run.result.trace.end(),result.trace.begin(),result.trace.end());
    run.result.messages.insert(run.result.messages.end(),result.messages.begin(),result.messages.end());
    run.result.outputs.insert(run.result.outputs.end(),result.outputs.begin(),result.outputs.end());
    std::vector<Tensor> hidden(c.batch,at::zeros({c.width},inputs.options()));
    for (const auto& value : result.outputs) hidden.at(value.batch) = value.value;
    logits.push_back(project(at::stack(hidden),f.head));
  }
  run.result.continuation = cursor.snapshot();
  if (candidate && resident) {
    for (const auto& [owner,s] : run.result.continuation.states) {
      const auto expected = placement.devices.at(placement.node_device.at(owner.second));
      if (s.value.device()!=expected) throw std::runtime_error("state is not resident on its shard");
      for (const auto& [name,t] : s.slots)
        if(t.device()!=expected) throw std::runtime_error("cache is not resident on its shard");
    }
    for(const auto& a : run.result.continuation.pending)
      if(a.value.device()!=placement.devices.at(placement.node_device.at(f.graph.edges.at(a.source).source)))
        throw std::runtime_error("pending message left its source device");
  }
  for (const auto& event : run.result.trace) if (event.batch == 0 && event.active && !event.emitted.empty()) {
    run.roots.push_back(event.full); run.roots.push_back(event.emitted.front().value); break;
  }
  if (!run.result.continuation.pending.empty()) run.roots.push_back(run.result.continuation.pending.front().value);
  const auto& first = run.result.continuation.states.begin()->second;
  run.roots.push_back(first.value);
  for (const auto& [name,value] : first.slots) run.roots.push_back(value);
  run.roots.push_back(at::stack(logits));
  if (candidate) synchronize(placement);
  return run;
}
}
void check(const pdg_scale::Config& c,const pdg_scale::Topology& topology,at::Device device,Index devices,const std::string& policy,bool resident,bool conditioned) {
  if (c.width > 64 || c.batch > 8 || c.steps > 6) throw std::invalid_argument("placement parity needs bounded small tensors");
  for (bool enable_grad : {false,true}) {
  at::AutoGradMode grad(enable_grad);
  auto expected = execute(c,topology,at::Device(at::kCPU),1,false,policy,false);
  auto actual = execute(c,topology,device,devices,true,policy,resident);
  tide_bench::compare(actual.result,expected.result,true,at::kFloat);
  if (actual.leaves.size()!=expected.leaves.size() || actual.roots.size()!=expected.roots.size())
    throw std::runtime_error("placement changed root/owner count");
  std::set<const c10::TensorImpl*> leaves;
  for (const auto& leaf : actual.leaves) leaves.insert(leaf.unsafeGetTensorImpl());
  if (leaves.size()!=actual.leaves.size()) throw std::runtime_error("placement duplicated parameter owners");
  for (size_t root=0;root<expected.roots.size();++root) {
    if (!at::allclose(actual.roots[root].to(at::kCPU),expected.roots[root],1e-5,1e-6))
      throw std::runtime_error("placement changed public root value");
    if (actual.roots[root].requires_grad()!=expected.roots[root].requires_grad())
      throw std::runtime_error("placement changed public root connectivity");
    if (!expected.roots[root].requires_grad()) continue;
    for (bool zero : {false,true})
      check_vjp(expected.roots[root],actual.roots[root],expected.leaves,actual.leaves,root,zero,conditioned);
  }
  }
  std::cout << "CHECK complete CPU scalar-slot vs placed packed-row values/routes/state/history/pending, logits, isolated VJPs and None/zero: passed\n" << std::flush;
}
}  // namespace accelerator_scale
