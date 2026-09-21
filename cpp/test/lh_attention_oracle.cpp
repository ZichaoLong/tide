// Actual original Attention/KVHidden inference; never adopt its custom backward.
#include "AccumulateLocal.h"
#include "portable_torch/runtime.hpp"
#include "tide/fiber_attention.h"
#include "tide/stream.h"
#include "tide/frontier.h"
#include <ATen/Parallel.h>
#include <iostream>
#include <tuple>

namespace {
namespace AL = AccumulateLocal;
using tide::Index;
using Mode = AL::KVHidden::AttentionMode;
using Key = std::tuple<Index, Index, Index>;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void close(const at::Tensor& a, const at::Tensor& b, const char* message, bool fp32_payload = false) {
  const bool fp64 = a.scalar_type() == at::kDouble && !fp32_payload;
  require(a.sizes() == b.sizes() && a.scalar_type() == b.scalar_type()
          && at::allclose(a, b, fp64 ? 1e-8 : 1e-5, fp64 ? 1e-10 : 1e-6), message);
}
std::map<std::string, at::Tensor> cache(const AL::KVHidden& h, const at::TensorOptions& opts) {
  at::Tensor k, v;
  if (AL::KVHidden::attention_mode == Mode::LOOP || AL::KVHidden::attention_mode == Mode::PACKED) {
    k = h.keys.empty() ? at::empty({h.n_head, 0, h.D/h.n_head}, opts) : at::cat(h.keys, 1);
    v = h.values.empty() ? at::empty({h.n_head, 0, h.D/h.n_head}, opts) : at::cat(h.values, 1);
  } else { k = h.keys_cache.slice(1, 0, h.endidx); v = h.values_cache.slice(1, 0, h.endidx); }
  return {{"key", k.transpose(0, 1).clone()}, {"value", v.transpose(0, 1).clone()},
          {"log_bias", h.total_growth_rate.slice(0, 0, h.endidx).clone()}};
}
std::map<std::string, at::Tensor> cache(const AL::BatchPtrKVHidden& h, Index b, const at::TensorOptions& opts) {
  if (AL::KVHidden::attention_mode != Mode::CROSSBATCH) return cache(*h.hptrs[b], opts);
  const auto n = h.endindices[b];
  return {{"key", h.batch_keys_cache[b].slice(1, 0, n).transpose(0, 1).clone()},
          {"value", h.batch_values_cache[b].slice(1, 0, n).transpose(0, 1).clone()},
          {"log_bias", h.batch_total_growth_rate[b].slice(0, 0, n).clone()}};
}
Index check_case(const at::TensorOptions& opts, Mode mode, bool multi, bool clear, double decay,
                 Index width, Index heads, bool bias, int schedule, const std::string& pool = "sum") {
  const Index nodes = 2, sources = 5, batches = 4, ticks = 24;
  const std::map<std::string, std::string> confluence{{"sum", "add"}, {"mean", "average"}, {"linear", "linear"},
    {"active-softmax", "actsoftmax"}, {"all-softmax", "allsoftmax"}};
  AL::KVHidden::attention_mode = mode; AL::KVHidden::multi_batch_forward_mode = multi;
  AL::KVHidden::is_no_grad = true;
  std::vector<std::shared_ptr<AL::Attention>> original;
  std::vector<std::shared_ptr<AL::BatchPtrKVHidden>> hidden;
  tide::Graph g; tide::Model m; tide::Continuation initial; initial.batch_size = batches;
  for (Index node = 0; node < nodes; ++node) {
    tide::Node spec{node, clear}; spec.memory = "lh-fiber-attention-"+pool+"-repeat-v1";
    spec.query_heads = spec.kv_heads = heads; spec.readout = "norm-fp64-v1";
    g.nodes.push_back(spec); g.regions.push_back({1});
    auto al = std::make_shared<AL::Attention>(sources, width, nlohmann::json{
      {"confluence", confluence.at(pool)}, {"decay_rate", decay}, {"n_head", heads}, {"bias", bias}, {"block_size", 2}});
    al->to(opts.dtype().toScalarType());
    al->c_attn->weight.copy_(at::sin(at::arange(3*width*width, opts).reshape({3*width, width})*.31+node)*.3);
    al->c_proj->weight.copy_(at::cos(at::arange(width*width, opts).reshape({width, width})*.23)*.2);
    if (bias) {
      al->c_attn->bias.copy_(at::arange(3*width, opts)/30-.1);
      al->c_proj->bias.copy_(at::arange(width, opts)/20-.2);
    }
    original.push_back(al);
    tide::NodeWeights w{at::zeros({width}, opts), at::zeros({width, width}, opts),
                        at::zeros({width}, opts), at::zeros({width}, opts)};
    w.extra = {{"fiber_qkv", al->c_attn->weight.t().clone()}, {"fiber_out", al->c_proj->weight.t().clone()},
               {"fiber_qkv_bias", bias ? al->c_attn->bias.clone() : at::zeros({3*width}, opts)},
               {"fiber_out_bias", bias ? al->c_proj->bias.clone() : at::zeros({width}, opts)},
               {"fiber_decay", at::scalar_tensor(decay, opts)}};
    auto pooling = al->ptr2confluence->named_parameters();
    if (pooling.contains("weight")) {
      pooling["weight"].copy_(at::arange(sources, opts)*.3-.4+node/10.);
      w.extra["fiber_pool"] = pooling["weight"].clone();
    }
    m.nodes.push_back(w);
    for (Index p = 0; p < sources; ++p) {
      g.inputs.push_back(node); m.input_scale.push_back(at::scalar_tensor(.7+p/10., opts));
    }
    auto bh = std::make_shared<AL::BatchPtrKVHidden>(batches, width, heads); bh->to(opts.dtype().toScalarType());
    hidden.push_back(bh);
    for (Index b = 0; b < batches; ++b) {
      const Index rows = b % 3;
      if (rows) {
        auto k = at::sin(at::arange(heads*rows*(width/heads), opts).reshape({heads, rows, width/heads})*.17+b);
        if (mode == Mode::CROSSBATCH) {
          auto samples = at::tensor(std::vector<Index>{b}, at::kLong);
          auto lengths = at::tensor(std::vector<Index>{rows}, at::kLong);
          bh->addkvcache(k.transpose(0, 1), (k*.3+.1).transpose(0, 1), samples, lengths,
                        samples, lengths, at::full({rows}, b, at::kLong));
          bh->batch_total_growth_rate[b].slice(0, 0, rows).fill_(-.13*b);
        } else {
          bh->hptrs[b]->addkvcache(k, k*.3+.1);
          bh->hptrs[b]->total_growth_rate.slice(0, 0, rows).fill_(-.13*b);
        }
      }
      initial.states[{b, node}] = {at::zeros({width}, opts), -1, 0, cache(*bh, b, opts)};
    }
  }
  g.compile(); initial.identity = g.identity;
  tide::Options options; options.workers = schedule ? 3 : 1; options.packed = schedule != 0;
  std::unique_ptr<tide::Streaming> streaming;
  std::unique_ptr<tide::Frontier> frontier;
  if (schedule == 2) frontier = std::make_unique<tide::Frontier>(g, m, options);
  else streaming = std::make_unique<tide::Streaming>(g, m, options);
  auto run = [&](const tide::Continuation& q, const std::vector<tide::External>& xs, Index stop) {
    return streaming ? streaming->run(q, xs, stop, stop) : frontier->run(q, xs, stop, stop);
  };
  std::map<tide::Owner, Index> positions;
  std::map<Key, tide::State> proposals;
  std::vector<tide::External> all_inputs;
  tide::Continuation q = initial; Index candidates = 0, max_rows = 0;
  for (Index time = 0; time < ticks; ++time) {
    std::vector<tide::External> external;
    size_t count = 0;
    for (Index node = 0; node < nodes; ++node) {
      std::vector<at::Tensor> rows, ids; Index atoms = 0;
      for (Index b = 0; b < batches; ++b) {
        std::vector<at::Tensor> values; std::vector<Index> present;
        if (time >= 2 && time < ticks-3 && time%5 != 4 && b != 3 && (time+b+node)%4 != 0) {
          for (Index p : {0, 2, 4}) {
            if ((time+b+p)%3 == 0) continue;
            auto x = (time+node+p)%7 == 0 ? at::zeros({width}, opts)
              : at::cos(at::arange(width, opts)*.3+(time+b+p)/10.);
            const auto port = node*sources+p;
            values.push_back(x*m.input_scale[port]); present.push_back(p); ++atoms;
            external.push_back({b, port, positions[{b, port}]++, time, x});
          }
        }
        rows.push_back(values.empty() ? at::empty({0, width}, opts) : at::stack(values));
        ids.push_back(at::tensor(present, at::kLong));
      }
      AL::PtrBatchCHALInput input;
      if (atoms) { input = std::make_shared<AL::BatchCHALInput>(sources, rows, ids); input->check_validity(); }
      auto output = original[node]->forward(input, hidden[node]);  // Decays even on idle ticks.
      require(output.defined() == bool(input), "original Attention absent-input mismatch");
      if (!input) continue;
      const auto samples = input->get_sampleids();
      for (Index row = 0; row < samples.numel(); ++row) {
        auto b = samples[row].item<Index>(); auto slots = cache(*hidden[node], b, opts);
        max_rows = std::max(max_rows, slots.at("key").size(0));
        proposals[{time, b, node}] = {output[row].clone(), time, 0, slots}; ++candidates; ++count;
      }
      if (clear) hidden[node]->clear(samples);
    }
    auto result = run(q, external, time+1); q = result.continuation;
    all_inputs.insert(all_inputs.end(), external.begin(), external.end());
    require(result.trace.size() == count, "Attention candidate presence mismatch");
    for (const auto& event : result.trace) {
      require(event.active, "singleton Attention candidate unexpectedly unselected");
      const auto& expected = proposals.at({time, event.batch, event.node});
      close(event.proposal, expected.value, "original Attention proposal mismatch");
      close(event.comparison, expected.value, "Attention clear erased comparison");
      // Read itself must be FP64. Cross-implementation error in its FP32 input
      // remains FP32 error after promotion; promotion cannot recover lost bits.
      close(event.descriptor, at::norm(event.proposal.to(at::kDouble)), "Attention Read is not an FP64 norm");
      close(event.descriptor, at::norm(expected.value.to(at::kDouble)), "Attention original norm mismatch",
            opts.dtype().toScalarType() == at::kFloat);
      for (const auto& [name, value] : expected.slots)
        close(event.proposed_state.slots.at(name), value, "original Attention proposal cache mismatch");
    }
    for (Index node = 0; node < nodes; ++node) for (Index b = 0; b < batches; ++b) {
      const auto& state = q.states.at({b, node}); const auto expected = cache(*hidden[node], b, opts);
      for (const auto& name : {"key", "value"}) close(state.slots.at(name), expected.at(name), "Attention cut cache mismatch");
      close(tide::decode_fiber_bias(m.nodes[node], state, time+1), expected.at("log_bias"), "Attention cut decoded bias mismatch");
    }
  }
  if (!clear) require(max_rows > AL::KVHidden::init_cache_size, "oracle did not exercise cache growth");
  auto whole = run(initial, all_inputs, ticks);
  require(whole.trace.size() == proposals.size(), "whole-window Attention candidate mismatch");
  for (const auto& event : whole.trace) {
    const auto& expected = proposals.at({event.time, event.batch, event.node});
    close(event.proposal, expected.value, "whole-window original Attention proposal mismatch");
    for (const auto& [name, value] : expected.slots)
      close(event.proposed_state.slots.at(name), value, "whole-window original Attention cache mismatch");
  }
  for (const auto& [owner, a] : whole.continuation.states) {
    const auto& b = q.states.at(owner);
    require(a.last_time == b.last_time && a.observations == b.observations, "Attention cut clock/count mismatch");
    close(a.value, b.value, "Attention cut encoded value mismatch");
    for (const auto& [name, value] : a.slots) close(value, b.slots.at(name), "Attention cut encoded cache mismatch");
  }
  return candidates;
}
#ifdef ENABLE_RUNTIME_ASSERTION
void check_original_fp64_assertion_limit(const at::TensorOptions& opts) {
  AL::ActSoftmaxConfluence original(3); original.to(at::kDouble);
  // Actual forward computes a double result, then its diagnostic denominator
  // calls SumCoe(num, indptr) with the original hardcoded float32 default.
  bool rejected = false;
  try {
    original.forward(at::ones({2, 4}, opts), at::tensor({0, 2}, at::kLong),
                     at::tensor({0, 2}, at::kLong), at::tensor({2}, at::kLong), {});
  } catch (const c10::Error& error) {
    rejected = std::string(error.what()).find("expected scalar type Float but found Double") != std::string::npos;
  }
  require(rejected, "original FP64 active-softmax assertion limitation changed; review oracle coverage");
}
#endif
}  // namespace
int main(int argc, char** argv) {
  try {
    const auto args = portable_torch::parse_cli(argc, argv);
    if (args.help) { portable_torch::print_usage(std::cout, argv[0]); return 0; }
    const auto device = portable_torch::resolve_device(args);
    if (!device.is_cpu() || (args.dtype != at::kDouble && args.dtype != at::kFloat))
      throw std::invalid_argument("LH Attention oracle requires CPU FP64/FP32");
    at::set_num_threads(1); at::NoGradGuard guard;
    const auto opts = at::TensorOptions().dtype(args.dtype).device(device);
    Index candidates = 0, cases = 0, unavailable = 0;
#ifdef ENABLE_RUNTIME_ASSERTION
    if (args.dtype == at::kDouble) check_original_fp64_assertion_limit(opts);
#endif
    for (auto mode : {Mode::LOOP, Mode::PACKED, Mode::CACHEDMATMUL, Mode::CACHEDPACKED, Mode::CACHEDATTENTION, Mode::CROSSBATCH})
      for (bool multi : {false, true}) for (bool clear : {false, true}) for (double decay : {0., .01})
        for (bool wide : {false, true}) for (int schedule = 0; schedule < 3; ++schedule) {
          if (mode == Mode::CROSSBATCH && !multi) continue;  // Original API forbids this combination.
          candidates += check_case(opts, mode, multi, clear, decay, wide ? 4 : 1, wide ? 2 : 1, wide, schedule); ++cases;
        }
    for (const auto& pool : {"mean", "linear", "active-softmax", "all-softmax"})
      for (auto mode : {Mode::PACKED, Mode::CROSSBATCH}) for (bool multi : {false, true})
        for (bool clear : {false, true}) for (int schedule = 0; schedule < 3; ++schedule) {
          if (mode == Mode::CROSSBATCH && !multi) continue;
#ifdef ENABLE_RUNTIME_ASSERTION
          if (args.dtype == at::kDouble && std::string(pool) == "active-softmax" && multi) {
            ++unavailable; continue;  // Separately qualified in the assertions-off executable.
          }
#endif
          candidates += check_case(opts, mode, multi, clear, .01, 4, 2, true, schedule, pool); ++cases;
        }
    std::cout << "original-LH-attention: passed; " << cases << " cases, " << cases*24 << " ticks, " << candidates
              << " candidate occurrences; LOOP/PACKED/CACHEDMATMUL/CACHEDPACKED/CACHEDATTENTION/CROSSBATCH; "
                 "single/multi projection; serial/packed/frontier; tick/whole windows; 264 sum + "
              << cases-264 << " mean/linear/softmax cases; unavailable original FP64 assertion cases=" << unavailable << '\n';
    return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 2; }
}
