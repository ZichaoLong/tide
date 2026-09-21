// Actual original Pronounce entry point with phase/sample gathering and token clock.
#include "CortexNet.h"
#include "lh_cache_oracle.h"
#include "portable_torch/runtime.hpp"
#include "tide/fiber_attention.h"
#include "tide/lazy_add.h"
#include "tide/stream.h"
#include "tide/frontier.h"
#include "tide/token_window.h"
#include <ATen/Parallel.h>
#include <iostream>
#include <set>

namespace {
namespace AL = AccumulateLocal;
using tide::Index;
using Mode = AL::KVHidden::AttentionMode;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void close(const at::Tensor& a, const at::Tensor& b, const char* message) {
  const bool fp64 = a.scalar_type() == at::kDouble;
  require(a.sizes() == b.sizes() && a.scalar_type() == b.scalar_type()
          && at::allclose(a, b, fp64 ? 1e-8 : 1e-5, fp64 ? 1e-10 : 1e-6), message);
}
tide::NodeWeights import_weights(Pronounce& original, const at::TensorOptions& opts,
                                const std::string& pool, Index layers, Index width, Index vocab, bool bias) {
  tide::NodeWeights w{at::zeros({width}, opts), at::zeros({width, width}, opts),
                      at::zeros({width}, opts), at::zeros({width}, opts)};
  require(ConfigurableModule::get_norm_type(original.config.cfg) == ModuleTypes::NormType::IDENTITY
          && original.norm.ptr()->named_parameters().is_empty(), "original Pronounce default norm policy changed");
  auto head = original.lm_head;
  head->weight.copy_(at::cos(at::arange(vocab*width, opts).reshape({vocab, width})*.23)*.3);
  // The first width vocabulary coordinates expose the pre-head output directly;
  // additional vocabulary coordinates exercise a nontrivial rectangular head.
  head->weight.slice(0, 0, width).copy_(at::eye(width, opts));
  if (bias) { head->bias.copy_(at::arange(vocab, opts)/20); head->bias.slice(0, 0, width).zero_(); }
  w.extra["token_head"] = head->weight.clone();
  w.extra["token_head_bias"] = bias ? head->bias.clone() : at::zeros({vocab}, opts);
  if (pool == "add") {
    w.extra["add_retention"] = at::scalar_tensor(.97, opts);
    return w;
  }
  // BaseAL owns the inherited forward method, so AnyModule::ptr<Attention>()
  // does not compile with this LibTorch API; cast the actual stored Module.
  auto attention = std::dynamic_pointer_cast<AL::Attention>(original.chal.ptr());
  require(static_cast<bool>(attention), "original Pronounce did not construct Attention");
  attention->c_attn->weight.copy_(.2+at::sin(at::arange(3*width*width, opts).reshape({3*width, width})*.17)*.1);
  attention->c_proj->weight.copy_(.3+at::cos(at::arange(width*width, opts).reshape({width, width})*.21)*.2);
  if (bias) { attention->c_attn->bias.fill_(.4); attention->c_proj->bias.fill_(.5); }
  w.extra["fiber_qkv"] = attention->c_attn->weight.t().clone();
  w.extra["fiber_out"] = attention->c_proj->weight.t().clone();
  w.extra["fiber_qkv_bias"] = bias ? attention->c_attn->bias.clone() : at::zeros({3*width}, opts);
  w.extra["fiber_out_bias"] = bias ? attention->c_proj->bias.clone() : at::zeros({width}, opts);
  w.extra["fiber_decay"] = at::scalar_tensor(.03, opts);
  auto coefficients = attention->ptr2confluence->named_parameters();
  if (coefficients.contains("weight")) {
    coefficients["weight"].copy_(at::arange(layers, opts)*.3-.4);
    w.extra["fiber_pool"] = coefficients["weight"].clone();
  }
  return w;
}
Index check_case(const at::TensorOptions& opts, const std::string& pool, Index layers, bool bias,
                 int original_mode, int schedule) {
  const Index batches = 4, width = 4, vocab = 7, tokens = 6;
  AL::TensorHidden::multi_batch_forward_mode = original_mode != 0; AL::TensorHidden::is_no_grad = true;
  AL::KVHidden::multi_batch_forward_mode = original_mode != 0; AL::KVHidden::is_no_grad = true;
  AL::KVHidden::attention_mode = original_mode == 2 ? Mode::CROSSBATCH : Mode::PACKED;
  const std::map<std::string, std::string> confluence{{"add", "add"}, {"sum", "add"}, {"mean", "average"},
    {"linear", "linear"}, {"active-softmax", "actsoftmax"}, {"all-softmax", "allsoftmax"}};
  Pronounce original(vocab, layers, width, nlohmann::json{{"chal", pool == "add" ? "add" : "attention"},
    {"confluence", confluence.at(pool)}, {"n_head", 2}, {"bias", bias}, {"decay_rate", .03}, {"block_size", 2}});
  original.to(opts.dtype().toScalarType());
  auto hidden = original.GenBatchPtrBaseHidden(batches, opts.dtype().toScalarType(), opts.device());
  auto weights = import_weights(original, opts, pool, layers, width, vocab, bias);
  tide::Graph g; g.nodes = {{0}};
  g.nodes[0].memory = pool == "add" ? "lh-add-repeat-v1" : "lh-fiber-attention-"+pool+"-repeat-v1";
  g.nodes[0].full = "lh-identity-identity-v1"; g.nodes[0].query_heads = g.nodes[0].kv_heads = 2;
  g.regions = {{1}}; g.inputs.assign(layers, 0); g.outputs = {0}; g.compile();
  tide::Model m; m.nodes = {weights}; m.input_scale.assign(layers, at::ones({}, opts));
  m.output_scale = {at::ones({}, opts)};
  tide::Continuation initial; initial.identity = g.identity; initial.batch_size = batches;
  tide::Options options; options.workers = schedule ? 3 : 1; options.packed = schedule != 0;
  std::unique_ptr<tide::Streaming> streaming;
  std::unique_ptr<tide::Frontier> frontier;
  if (schedule == 2) frontier = std::make_unique<tide::Frontier>(g, m, options);
  else streaming = std::make_unique<tide::Streaming>(g, m, options);
  auto run = [&](const tide::Continuation& q, const std::vector<tide::External>& xs, Index stop) {
    return streaming ? streaming->run(q, xs, stop, stop) : frontier->run(q, xs, stop, stop);
  };
  auto empty = pool == "add" ? tide::State{at::zeros({width}, opts)}
                            : tide::make_fiber_attention_kernel(g.nodes[0], layers)->initial(weights);
  std::map<tide::Owner, at::Tensor> expected_logits;
  std::vector<tide::Output> all_body;
  auto q = initial; Index occurrences = 0;
  for (Index token = 0; token < tokens; ++token) {
    VPtrBatchSignals phases(layers); std::vector<tide::Output> body;
    std::set<Index> active;
    for (Index phase = 0; phase < layers; ++phase) {
      std::vector<at::Tensor> rows; std::vector<Index> samples;
      for (Index b = 0; b < batches-1; ++b) {
        if ((b == 1 && (token == 2 || token == 3)) || (b == 2 && token == 0)) continue;
        if ((token+b+phase)%3 == 1 && !(b == 0 && phase == 0)) continue;
        auto x = .7+at::sin(at::arange(width, opts)*.2+token*.1+phase*.13+b*.07)*.2;
        if (token == 4 && phase == 0 && b == 0) x = at::zeros({width}, opts);
        rows.push_back(x); samples.push_back(b); active.insert(b);
        body.push_back({b, token*layers+phase, 0, x});
      }
      if (!rows.empty()) phases[phase] = std::make_shared<BatchSignals>(at::stack(rows), at::tensor(samples, at::kLong));
    }
    auto logits = original.forward(batches, phases, hidden);  // Actual original phase gather + state + norm + head.
    require(logits.sizes() == at::IntArrayRef({static_cast<Index>(active.size()), vocab}), "Pronounce sample rows mismatch");
    Index row = 0;
    for (auto b : active) expected_logits[{token, b}] = logits[row++].clone();
    auto external = tide::token_inputs(body, q, layers, token+1, (token+1)*layers);
    auto result = run(q, external, token+1); q = result.continuation;
    all_body.insert(all_body.end(), body.begin(), body.end());
    require(result.outputs.size() == active.size(), "Pronounce candidate/output presence mismatch");
    occurrences += active.size();
    for (const auto& o : result.outputs) {
      require(o.time == token && o.port == 0 && active.count(o.batch), "Pronounce output coordinate mismatch");
      const auto& target = expected_logits.at({token, o.batch});
      close(o.value, target.slice(0, 0, width), "Pronounce normalized output mismatch");
      close(at::linear(o.value, weights.extra.at("token_head"), weights.extra.at("token_head_bias")), target,
            "Pronounce vocabulary projection mismatch");
    }
    for (Index b = 0; b < batches; ++b) {
      const auto found = q.states.find({b, 0}); const auto& state = found == q.states.end() ? empty : found->second;
      if (pool == "add") {
        auto& h = dynamic_cast<AL::BatchPtrTensorHidden&>(*hidden);
        close(tide::decode_add_repeat(weights, state, token+1), original_mode ? h.cache[b] : h.hptrs[b]->data,
              "Pronounce Add token-clock hidden mismatch");
      } else {
        auto slots = lh_oracle::cache(dynamic_cast<AL::BatchPtrKVHidden&>(*hidden), b, opts);
        for (const auto& name : {"key", "value"}) close(state.slots.at(name), slots.at(name), "Pronounce KV mismatch");
        close(tide::decode_fiber_bias(weights, state, token+1), slots.at("log_bias"), "Pronounce token-clock bias mismatch");
      }
    }
  }
  auto whole = run(initial, tide::token_inputs(all_body, initial, layers, tokens, tokens*layers), tokens);
  require(whole.outputs.size() == expected_logits.size() && whole.continuation.ledger == q.ledger,
          "Pronounce whole-window output/ledger mismatch");
  for (const auto& o : whole.outputs)
    close(at::linear(o.value, weights.extra.at("token_head"), weights.extra.at("token_head_bias")),
          expected_logits.at({o.time, o.batch}), "Pronounce whole-window logits mismatch");
  for (const auto& [owner, state] : q.states) {
    const auto& other = whole.continuation.states.at(owner);
    require(state.last_time == other.last_time && state.observations == other.observations, "Pronounce clock/count mismatch");
    close(state.value, other.value, "Pronounce encoded state mismatch");
    for (const auto& [name, value] : state.slots) close(value, other.slots.at(name), "Pronounce encoded slot mismatch");
  }
  return occurrences;
}
}  // namespace
int main(int argc, char** argv) {
  try {
    const auto args = portable_torch::parse_cli(argc, argv);
    if (args.help) { portable_torch::print_usage(std::cout, argv[0]); return 0; }
    const auto device = portable_torch::resolve_device(args);
    if (!device.is_cpu() || (args.dtype != at::kFloat && args.dtype != at::kDouble))
      throw std::invalid_argument("Pronounce oracle requires CPU FP64/FP32");
    at::set_num_threads(1); at::NoGradGuard guard;
    const auto opts = at::TensorOptions().dtype(args.dtype).device(device);
    Index cases = 0, unavailable = 0, occurrences = 0;
    for (const auto& pool : {"add", "sum", "mean", "linear", "active-softmax", "all-softmax"})
      for (Index layers : {1, 3}) for (bool bias : {false, true})
        for (int mode = 0; mode < 3; ++mode) for (int schedule = 0; schedule < 3; ++schedule) {
          if (std::string(pool) == "add" && mode == 2) continue;
#ifdef ENABLE_RUNTIME_ASSERTION
          if (args.dtype == at::kDouble && std::string(pool) == "active-softmax" && mode) { ++unavailable; continue; }
#endif
          try { occurrences += check_case(opts, pool, layers, bias, mode, schedule); }
          catch (const std::exception& e) {
            throw std::runtime_error(std::string("Pronounce pool=")+pool+" layers="+std::to_string(layers)
              +" bias="+std::to_string(bias)+" original_mode="+std::to_string(mode)+" schedule="+std::to_string(schedule)
              +": "+e.what());
          }
          ++cases;
        }
    std::cout << "original-LH-pronounce: passed; " << cases << " cases, " << cases*6 << " tokens, " << occurrences
              << " sample outputs; Add + five attention pools; phase gathering, identity norm, rectangular head and state; "
                 "unavailable original FP64 assertion cases=" << unavailable << '\n';
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 2; }
}
