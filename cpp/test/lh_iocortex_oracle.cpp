// Actual original IOCortexNet execution, mapped to two explicit graph clocks.
#include "lh_single_graph.h"
#include "portable_torch/runtime.hpp"
#include "tide/stream.h"
#include "tide/frontier.h"
#include "tide/token_window.h"
#include <ATen/Parallel.h>
#include <iostream>
#include <set>
#include <filesystem>

namespace {
using namespace lh_iocortex;
namespace AL = AccumulateLocal;
struct Counts { Index candidates = 0, messages = 0, outputs = 0, single_cuts = 0; };
tide::Continuation initial(const tide::Graph& g) {
  tide::Continuation q; q.identity = g.identity; q.batch_size = Fixture::batch; return q;
}
void check_logits(const Fixture& f, const tide::Result& r, Index token, const at::Tensor& target) {
  require(r.outputs.size() == Fixture::batch && target.size(0) == Fixture::batch, "IOCortex token sample presence mismatch");
  std::set<Index> seen;
  const auto& w = f.read_model.nodes[0];
  for (const auto& o : r.outputs) {
    require(o.time == token && o.port == 0 && o.batch >= 0 && o.batch < Fixture::batch && seen.insert(o.batch).second,
            "IOCortex token labels mismatch");
    close(at::linear(o.value, w.extra.at("token_head"), w.extra.at("token_head_bias")), target[o.batch],
          "IOCortex token logits mismatch");
  }
}
Counts check_tokens(const Fixture& f, int schedule, const std::string& export_path) {
  const Index tokens = 4, layers = f.policy.layers();
  VPtrBatchSignals iacts, oacts; VPtrBatchPtrBaseHidden ih, oh; PtrBatchPtrBaseHidden ph;
  ObservedSelector is(Fixture::batch, f.cfg, f.policy, ih, f.opts), os(Fixture::batch, f.cfg, f.policy, oh, f.opts);
  tide::Options options; options.workers = schedule ? 3 : 1; options.packed = schedule == 2;
  tide::Streaming body(f.body, f.model, options), read(f.readout, f.read_model, options);
  tide::Frontier frontier(f.readout, f.read_model, options);
  auto read_run = [&](const tide::Continuation& q, const std::vector<tide::External>& x, Index stop) {
    return schedule == 2 ? frontier.run(q, x, stop, stop) : read.run(q, x, stop, stop);
  };
  auto q = initial(f.body), rq = initial(f.readout); Counts counts;
  std::vector<tide::External> all_inputs; std::vector<at::Tensor> targets;
  at::Tensor feedback;
  for (Index token = 0; token < tokens; ++token) {
    auto ids = token >= 2 ? feedback : at::remainder(at::arange(Fixture::batch, at::kLong)+token*2, Fixture::vocab);
    // Original entry point does embedding, all L ticks, phase gather and readout.
    auto logits = f.original->think(ids, is, os, iacts, oacts, ih, oh, ph);
    targets.push_back(logits.clone());
    auto embedding = f.original->wte->forward(ids); std::vector<tide::External> inputs;
    for (Index b = 0; b < Fixture::batch; ++b) inputs.push_back({b, 0, token, token*layers, embedding[b]});
    auto r = body.run(q, inputs, (token+1)*layers, (token+1)*layers);
    compare_trace(f, r, is, os, q.cut); q = r.continuation;
    compare_states(f, q, ih, oh); compare_counts(f, q, is, os); compare_pending(f, q, iacts, oacts);
    auto readout = read_run(rq, tide::token_inputs(r.outputs, rq, layers, token+1, q.cut), token+1);
    check_logits(f, readout, token, logits); rq = readout.continuation;
    for (Index b = 0; b < Fixture::batch; ++b)
      compare_cache(f.read_model.nodes[0], rq.states.at({b, 0}), token+1, original_cache(ph, b, f.opts), "readout");
    // Finish readout before producing/sealing the next autoregressive token.
    feedback = at::argmax(logits, -1);
    const auto& w = f.read_model.nodes[0];
    for (const auto& o : readout.outputs)
      require(at::argmax(at::linear(o.value, w.extra.at("token_head"), w.extra.at("token_head_bias"))).item<Index>()
                == feedback[o.batch].item<Index>(), "IOCortex greedy feedback mismatch");
    all_inputs.insert(all_inputs.end(), inputs.begin(), inputs.end());
    counts.candidates += r.trace.size(); counts.messages += r.messages.size(); counts.outputs += readout.outputs.size();
  }
  auto whole = body.run(initial(f.body), all_inputs, tokens*layers, tokens*layers);
  compare_trace(f, whole, is, os, 0); compare_continuation(q, whole.continuation);
  auto all_read = read_run(initial(f.readout), tide::token_inputs(whole.outputs, initial(f.readout), layers, tokens, tokens*layers), tokens);
  compare_continuation(rq, all_read.continuation);
  for (Index token = 0; token < tokens; ++token) {
    tide::Result r; for (const auto& o : all_read.outputs) if (o.time == token) r.outputs.push_back(o);
    check_logits(f, r, token, targets[token]);
  }
  write_fixture(export_path, f, all_inputs, is, os, ih, oh, ph, targets);
  counts.single_cuts = check_single_graph(f, schedule, all_inputs, is, os, targets);
  return counts;
}
Counts check_ragged(const Fixture& f, int schedule, const std::string& export_path) {
  const Index ticks = 10;
  VPtrBatchSignals iacts(f.n), oacts(f.n);
  auto ih = f.original->inet->GenVPtrBatchPtrBaseHidden(Fixture::batch);
  auto oh = f.original->onet->GenVPtrBatchPtrBaseHidden(Fixture::batch);
  for (auto& h : ih) h->to(f.opts.dtype().toScalarType());
  for (auto& h : oh) h->to(f.opts.dtype().toScalarType());
  ObservedSelector is(Fixture::batch, f.cfg, f.policy, ih, f.opts), os(Fixture::batch, f.cfg, f.policy, oh, f.opts);
  tide::Options options; options.workers = schedule ? 3 : 1; options.packed = schedule == 2;
  tide::Streaming body(f.body, f.model, options);
  auto q = initial(f.body); std::vector<tide::External> all_inputs;
  std::map<Index, Index> positions; Counts counts;
  for (Index t = 0; t < ticks; ++t) {
    std::vector<Index> samples; std::vector<at::Tensor> rows; std::vector<tide::External> inputs;
    if (t == 2 || t == 5 || t == 8) for (Index b = 0; b < 3; ++b) {
      if ((b == 2 && t == 2) || (b == 1 && t == 5)) continue;  // Sample 3 always absent.
      auto x = .5+at::sin(at::arange(Fixture::width, f.opts)*.3+b*.2+t*.1)*.2;
      if (b == 1 && t == 2) x = at::zeros_like(x);
      samples.push_back(b); rows.push_back(x); inputs.push_back({b, 0, positions[b]++, t, x});
    }
    OVBatchExtraInput extra;
    if (!rows.empty()) {
      VBatchExtraInput values(f.n);
      values[0] = {std::make_shared<BatchSignals>(at::stack(rows), at::tensor(samples, at::kLong))}; extra = values;
    }
    std::tie(iacts, oacts) = f.original->think_single_step(Fixture::batch, iacts, oacts, extra, ih, oh, is, os);
    auto r = body.run(q, inputs, t+1, t+1); compare_trace(f, r, is, os, t); q = r.continuation;
    compare_states(f, q, ih, oh); compare_counts(f, q, is, os); compare_pending(f, q, iacts, oacts);
    all_inputs.insert(all_inputs.end(), inputs.begin(), inputs.end());
    counts.candidates += r.trace.size(); counts.messages += r.messages.size();
  }
  auto whole = body.run(initial(f.body), all_inputs, ticks, ticks);
  compare_trace(f, whole, is, os, 0); compare_continuation(q, whole.continuation);
  for (const auto& [owner, state] : q.states) require(owner.first != 3, "IOCortex fabricated absent sample state");
  write_fixture(export_path, f, all_inputs, is, os, ih, oh, nullptr, {});
  counts.single_cuts = check_single_graph(f, schedule, all_inputs, is, os);
  return counts;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const auto args = portable_torch::parse_cli(argc, argv);
    if (args.help) { portable_torch::print_usage(std::cout, argv[0]); return 0; }
    const auto device = portable_torch::resolve_device(args);
    if (!device.is_cpu() || (args.dtype != at::kFloat && args.dtype != at::kDouble))
      throw std::invalid_argument("IOCortex oracle requires CPU FP64/FP32");
    at::set_num_threads(1); at::NoGradGuard guard;
    const auto opts = at::TensorOptions().dtype(args.dtype).device(device);
    if (!args.output_dir.empty()) require(std::filesystem::create_directory(args.output_dir), "IOCortex fixture directory must be new");
    Index cases = 0, unavailable = 0; Counts totals;
    for (const auto& pool : {"add", "sum", "mean", "linear", "active-softmax", "all-softmax"})
      for (bool clear : {false, true}) for (bool lead : {false, true})
        for (int mode = 0; mode < 3; ++mode) for (int schedule = 0; schedule < 3; ++schedule) {
          if (std::string(pool) == "add" && mode == 2) continue;
#ifdef ENABLE_RUNTIME_ASSERTION
          if (args.dtype == at::kDouble && std::string(pool) == "active-softmax" && mode) { ++unavailable; continue; }
#endif
          try {
            AL::TensorHidden::multi_batch_forward_mode = mode != 0; AL::TensorHidden::is_no_grad = true;
            AL::KVHidden::multi_batch_forward_mode = mode != 0; AL::KVHidden::is_no_grad = true;
            AL::KVHidden::attention_mode = mode == 2 ? AL::KVHidden::AttentionMode::CROSSBATCH : AL::KVHidden::AttentionMode::PACKED;
            Fixture fixture(opts, {pool, clear, lead, mode});
            auto path = !args.output_dir.empty() && mode == 0 && schedule == 0 && lead
              ? args.output_dir+"/"+pool+"-clear"+std::to_string(clear) : std::string();
            auto a = check_tokens(fixture, schedule, path.empty() ? "" : path+"-tokens.json");
            auto b = check_ragged(fixture, schedule, path.empty() ? "" : path+"-ragged.json");
            totals.candidates += a.candidates+b.candidates; totals.messages += a.messages+b.messages; totals.outputs += a.outputs;
            totals.single_cuts += a.single_cuts+b.single_cuts;
          } catch (const std::exception& e) {
            throw std::runtime_error(std::string("IOCortex pool=")+pool+" clear="+std::to_string(clear)
                +" lead="+std::to_string(lead)+" original_mode="+std::to_string(mode)+" schedule="+std::to_string(schedule)+": "+e.what());
          }
          ++cases;
          if (cases%24 == 0) std::cout << "IOCortex progress: " << cases << " numerical cases passed\n" << std::flush;
        }
    std::cout << "original-LH-iocortex: passed; " << cases << " cases, " << totals.candidates << " candidates, "
              << totals.messages << " messages, " << totals.outputs << " token/sample logits; actual think + ragged ticks; "
                 "serial/parallel/packed, cuts, full hidden/history/pending; single-PDG cuts=" << totals.single_cuts
              << "; unavailable original FP64 assertion cases=" << unavailable << '\n';
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 2; }
}
