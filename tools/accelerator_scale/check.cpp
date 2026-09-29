#include "execution.h"
#include "check_vjp.h"
#include "../../cpp/bench/streaming.h"
#include <algorithm>
#include <iostream>
#include <set>

namespace accelerator_scale {
namespace {
struct Run { Result result; std::vector<Tensor> leaves, roots; };
Run execute(pdg_scale::Config c, const pdg_scale::Topology& topology, at::Device device,
            Index devices, bool candidate, const std::string& policy, bool resident, const Scoring& scoring,const std::string& ranking_device="cpu",const std::string& event_device="cpu") {
  portable_torch::seed_runtime(at::Device(at::kCPU), c.runtime.seed);
  c.emission = candidate ? "row" : "slot";
  if(!candidate && c.runtime.dtype==at::kHalf && c.reference_float32) {
    c.runtime.dtype=at::kFloat;c.quantized_fp16_reference=true;
  }
  auto f = pdg_scale::fixture(c,topology);
  configure_scoring(f,scoring);
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
  placement.scoring = scoring;placement.ranking_device=ranking_device;placement.event_device=event_device;
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
    auto payload=[&](const Tensor& value) {
      if(value.defined() && value.scalar_type()!=c.runtime.dtype)
        throw std::runtime_error("resident payload changed configured dtype");
    };
    for(const auto& owner:f.owners)payload(owner);
    for(const auto& value:logits)payload(value);
    for(const auto& output:run.result.outputs)payload(output.value);
    for(const auto& a:run.result.messages)payload(a.value);
    for (const auto& e : run.result.trace) {
      for(const auto& value:{e.content,e.proposal,e.comparison,e.next,e.full})payload(value);
      if(e.control.scalar_type()!=at::kFloat)throw std::runtime_error("consumer control changed FP32 policy");
      const auto node_device = placement.devices.at(placement.node_device.at(e.node));
      const auto expected_read = scoring.read_device == "cpu" ? at::Device(at::kCPU) : node_device;
      if (e.descriptor.device() != expected_read) throw std::runtime_error("Read left its configured device");
      const auto dtype = f.graph.nodes[e.node].readout == "linear-v1" ? c.runtime.dtype : scoring.dtype;
      if (e.descriptor.scalar_type() != dtype) throw std::runtime_error("Read changed configured precision");
      if (e.control.device() != node_device) throw std::runtime_error("control did not reach node device");
    }
    for (const auto& [owner,s] : run.result.continuation.states) {
      const auto expected = placement.devices.at(placement.node_device.at(owner.second));
      payload(s.value);for(const auto& [name,t]:s.slots)payload(t);
      if (s.value.device()!=expected) throw std::runtime_error("state is not resident on its shard");
      for (const auto& [name,t] : s.slots)
        if(t.device()!=expected) throw std::runtime_error("cache is not resident on its shard");
    }
    for(const auto& a : run.result.continuation.pending) {
      payload(a.value);
      if(a.value.device()!=placement.devices.at(placement.node_device.at(f.graph.edges.at(a.source).source)))
        throw std::runtime_error("pending message left its source device");
    }
  }
  for (const auto& event : run.result.trace) if (event.batch == 0 && event.active && !event.emitted.empty()) {
    run.roots.push_back(event.full); run.roots.push_back(event.emitted.front().value);
    run.roots.push_back(event.descriptor); run.roots.push_back(event.control); break;
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
void check(const pdg_scale::Config& c,const pdg_scale::Topology& topology,at::Device device,Index devices,const std::string& policy,bool resident,bool conditioned,const Scoring& scoring,bool reference_fp64,const std::string& ranking_device,const std::string& event_device) {
  if (c.width > 64 || c.batch > 8 || c.steps > 6) throw std::invalid_argument("placement parity needs bounded small tensors");
  for (bool enable_grad : {false,true}) {
  at::AutoGradMode grad(enable_grad);
  Scoring reference; reference.dtype = reference_fp64 ? at::kDouble : scoring.dtype;
  auto expected = execute(c,topology,at::Device(at::kCPU),1,false,policy,false,reference);
  auto actual = execute(c,topology,device,devices,true,policy,resident,scoring,ranking_device,event_device);
  auto comparable = actual.result;
  size_t route_mismatches=0;double score_max_abs=0;
  if (actual.result.trace.size()!=expected.result.trace.size())
    throw std::runtime_error("precision/backend comparison changed event inventory");
  for (size_t i=0;i<comparable.trace.size();++i) {
    auto& a=comparable.trace[i];const auto& b=expected.result.trace[i];
    route_mismatches += std::tie(a.batch,a.node,a.time,a.active) != std::tie(b.batch,b.node,b.time,b.active);
    score_max_abs=std::max(score_max_abs,(a.descriptor.detach().to(at::kCPU).to(at::kDouble)-b.descriptor.to(at::kDouble)).abs().item<double>());
    // Cross-precision comparison is explicitly requested; only descriptor
    // metadata differs intentionally. Routes and every other observable stay strict.
    a.descriptor=a.descriptor.to(at::kCPU).to(b.descriptor.scalar_type());
  }
  std::cout << "CHECK scoring read=" << scoring.read_device << '/' << scoring.dtype_name()
            << " controls=" << scoring.control_device << " reference=cpu/" << reference.dtype_name()
            << " payload=" << portable_torch::dtype_name(c.runtime.dtype) << " atol=" << c.check_atol << " rtol=" << c.check_rtol
            << " score_max_abs=" << score_max_abs << " route_mismatches=" << route_mismatches << '\n' << std::flush;
  tide_bench::compare(comparable,expected.result,true,c.runtime.dtype,std::nullopt,c.check_rtol,c.check_atol);
  if (actual.leaves.size()!=expected.leaves.size() || actual.roots.size()!=expected.roots.size())
    throw std::runtime_error("placement changed root/owner count");
  std::set<const c10::TensorImpl*> leaves;
  for (const auto& leaf : actual.leaves) leaves.insert(leaf.unsafeGetTensorImpl());
  if (leaves.size()!=actual.leaves.size()) throw std::runtime_error("placement duplicated parameter owners");
  for (size_t root=0;root<expected.roots.size();++root) {
    if (!at::allclose(actual.roots[root].to(at::kCPU).to(expected.roots[root].scalar_type()),expected.roots[root],c.check_rtol,c.check_atol))
      throw std::runtime_error("placement changed public root value");
    if (actual.roots[root].requires_grad()!=expected.roots[root].requires_grad())
      throw std::runtime_error("placement changed public root connectivity");
    if (!expected.roots[root].requires_grad()) continue;
    for (bool zero : {false,true})
      check_vjp(expected.roots[root],actual.roots[root],expected.leaves,actual.leaves,root,zero,conditioned,c.check_rtol,c.check_atol,c.runtime.dtype==at::kHalf);
  }
  }
  std::cout << "CHECK complete CPU scalar-slot vs placed packed-row values/routes/state/history/pending, logits, isolated VJPs and None/zero: passed\n" << std::flush;
}
}  // namespace accelerator_scale
