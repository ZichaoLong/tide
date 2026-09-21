// Untouched original Add/Hidden/Confluence sources are linked from the snapshot.
#include "AccumulateLocal.h"
#include "portable_torch/runtime.hpp"
#include "tide/lazy_add.h"
#include "tide/stream.h"
#include "tide/frontier.h"
#include <ATen/Parallel.h>
#include <iostream>
#include <tuple>

namespace {
namespace AL = AccumulateLocal;
using tide::Index;
using Key = std::tuple<Index, Index, Index>;  // tick, sample, node
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void close(const at::Tensor& a, const at::Tensor& b, const char* message) {
  const bool fp64 = a.scalar_type() == at::kDouble;
  require(a.sizes() == b.sizes() && at::allclose(a, b, fp64 ? 1e-8 : 1e-5, fp64 ? 1e-10 : 1e-6), message);
}
Index check_case(const at::TensorOptions& opts, int hidden_mode, bool clear, double decay, int schedule) {
  const Index nodes = 3, sources = 5, batches = 4, width = 3, ticks = 24;
  AL::TensorHidden::multi_batch_forward_mode = hidden_mode != 0;
  AL::TensorHidden::is_no_grad = hidden_mode != 2;
  std::vector<std::shared_ptr<AL::Add>> original;
  std::vector<std::shared_ptr<AL::BatchPtrTensorHidden>> hidden;
  tide::Graph g; tide::Model m; tide::Continuation initial; initial.batch_size = batches;
  for (Index v = 0; v < nodes; ++v) {
    tide::Node node{v, clear}; node.memory = "lh-add-repeat-v1"; node.readout = "norm-fp64-v1";
    g.nodes.push_back(node); g.regions.push_back({1});
    tide::NodeWeights w{at::zeros({width}, opts), at::zeros({width, width}, opts),
                        at::zeros({width}, opts), at::zeros({width}, opts)};
    w.extra["add_retention"] = at::scalar_tensor(1.0-decay, opts); m.nodes.push_back(w);
    for (Index p = 0; p < sources; ++p) { g.inputs.push_back(v); m.input_scale.push_back(at::ones({}, opts)); }
    original.push_back(std::make_shared<AL::Add>(sources, width,
                       nlohmann::json{{"confluence", "add"}, {"decay_rate", decay}}));
    original.back()->to(opts.dtype().toScalarType());
    hidden.push_back(std::make_shared<AL::BatchPtrTensorHidden>(batches, width, opts));
    for (Index b = 0; b < batches; ++b) {
      auto value = at::tensor(std::vector<double>{.3+b/8., -.2-v/8., .17}, opts);
      initial.states[{b, v}] = {value.clone()};
      if (hidden_mode != 0) hidden.back()->cache[b].copy_(value);
      if (hidden_mode != 1) hidden.back()->hptrs[b]->data = value.clone();
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
  auto physical = [&](Index b, Index v) {
    return hidden_mode ? hidden[v]->cache[b] : hidden[v]->hptrs[b]->data;
  };
  std::map<tide::Owner, Index> positions;
  std::map<Key, at::Tensor> proposals;
  std::vector<tide::External> all_inputs;
  tide::Continuation q = initial; Index candidates = 0;
  for (Index time = 0; time < ticks; ++time) {
    std::vector<tide::External> external;
    for (Index v = 0; v < nodes; ++v) {
      std::vector<at::Tensor> rows, ids;
      Index atom_count = 0;
      for (Index b = 0; b < batches; ++b) {
        std::vector<at::Tensor> values; std::vector<Index> sources_present;
        if (time >= 2 && time < ticks-3 && time % 5 != 4 && b != 3 && (time+2*b+v)%4 != 0) {
          for (Index p : {0, 2, 4}) {
            if ((time+b+p)%3 == 0) continue;
            // A numerical zero is still a real source occurrence.
            auto x = (time+v+p)%7 == 0 ? at::zeros({width}, opts)
              : at::tensor(std::vector<double>{(time+b+1)/32., (p-v)/16., -.125}, opts);
            values.push_back(x); sources_present.push_back(p); ++atom_count;
            const auto port = v*sources+p;
            external.push_back({b, port, positions[{b, port}]++, time, x});
          }
        }
        rows.push_back(values.empty() ? at::empty({0, width}, opts) : at::stack(values));
        ids.push_back(at::tensor(sources_present, at::kLong));
      }
      AL::PtrBatchCHALInput input;
      if (atom_count) { input = std::make_shared<AL::BatchCHALInput>(sources, rows, ids); input->check_validity(); }
      auto output = original[v]->forward(input, hidden[v]);  // Includes decay on every tick, even null input.
      require(output.defined() == bool(input), "original Add changed absent-input behavior");
      if (!input) continue;
      const auto samples = input->get_sampleids();
      for (Index row = 0; row < samples.numel(); ++row) {
        proposals[{time, samples[row].item<Index>(), v}] = output[row].clone(); ++candidates;
      }
      if (clear) hidden[v]->clear(samples);
    }
    auto result = run(q, external, time+1); q = result.continuation;
    all_inputs.insert(all_inputs.end(), external.begin(), external.end());
    size_t expected_count = 0;
    for (const auto& [key, _] : proposals) if (std::get<0>(key) == time) ++expected_count;
    require(result.trace.size() == expected_count, "Add candidate presence mismatch");
    for (const auto& e : result.trace) {
      require(e.active, "singleton Add candidate unexpectedly unselected");
      const auto& y = proposals.at({time, e.batch, e.node});
      close(e.proposal, y, "Tide/original Add pre-clear proposal mismatch");
      close(e.comparison, y, "Add clear erased comparison snapshot");
      close(e.descriptor, at::norm(y.to(at::kDouble)), "Add norm Read mismatch");
    }
    for (Index v = 0; v < nodes; ++v) for (Index b = 0; b < batches; ++b) {
      close(tide::decode_add_repeat(m.nodes[v], q.states.at({b, v}), time+1), physical(b, v),
            "Tide lazy cut/original eager Add hidden mismatch");
      if (hidden_mode == 2) close(hidden[v]->hptrs[b]->data, physical(b, v), "original cached/individual hidden mismatch");
    }
  }
  // A whole frontier window can precompute sequences. Its values must also match
  // the original tick loop, not only the one-tick continuation invocation above.
  auto whole = run(initial, all_inputs, ticks);
  require(whole.trace.size() == proposals.size(), "whole-window Add candidate mismatch");
  for (const auto& e : whole.trace) close(e.proposal, proposals.at({e.time, e.batch, e.node}), "whole-window Add proposal mismatch");
  for (Index v = 0; v < nodes; ++v) for (Index b = 0; b < batches; ++b) {
    const auto& a = whole.continuation.states.at({b, v}); const auto& bstate = q.states.at({b, v});
    require(a.last_time == bstate.last_time && a.observations == bstate.observations, "Add cut clock/count mismatch");
    close(a.value, bstate.value, "Add cut encoded state mismatch");
    close(tide::decode_add_repeat(m.nodes[v], a, ticks), physical(b, v), "Add final decoded hidden mismatch");
  }
  return candidates;
}
}  // namespace
int main(int argc, char** argv) {
  try {
    const auto args = portable_torch::parse_cli(argc, argv);
    if (args.help) { portable_torch::print_usage(std::cout, argv[0]); return 0; }
    const auto device = portable_torch::resolve_device(args);
    if (!device.is_cpu() || (args.dtype != at::kDouble && args.dtype != at::kFloat))
      throw std::invalid_argument("LH Add oracle requires CPU FP64/FP32");
    at::set_num_threads(1); at::NoGradGuard guard;
    const auto opts = at::TensorOptions().dtype(args.dtype).device(device);
    Index candidates = 0;
    for (int hidden = 0; hidden < 3; ++hidden) for (bool clear : {false, true})
      for (double decay : {0., .01, 1.}) for (int schedule = 0; schedule < 3; ++schedule)
        candidates += check_case(opts, hidden, clear, decay, schedule);
    std::cout << "original-LH-add: passed; 54 cases, 1296 ticks, " << candidates
              << " candidate occurrences; single/cached/dual hidden; serial/packed/frontier; tick/whole windows\n";
    return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 2; }
}
