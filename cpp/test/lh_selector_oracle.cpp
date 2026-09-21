// Original LH Selector.cpp is compiled unchanged from a hashed external snapshot.
#include "Selector.h"
#include "portable_torch/runtime.hpp"
#include "tide/stream.h"
#include "tide/frontier.h"
#include <ATen/Parallel.h>
#include <iostream>
#include <map>
#include <stdexcept>

namespace {
using tide::Index;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
Index count(const tide::Continuation& q, Index batch, Index region, const std::string& field, Index node) {
  auto h = q.history.find({batch, region});
  if (h == q.history.end()) return 0;
  const auto& values = h->second.node_maps.at(field); auto it = values.find(node);
  return it == values.end() ? 0 : it->second;
}
Index check_case(const at::TensorOptions& opts, bool lead, Index budget, int schedule) {
  const Index batch_size = 4, ticks = 24;
  auto cfg = std::make_shared<GraphConfig>(nlohmann::json::parse(
      R"({"levelnum":2,"localnum":4,"base_num_or_hpnums":[0,1,2,8]})"));
  NaiveSelector heap(batch_size, budget, cfg, lead, false), tensor(batch_size, budget, cfg, lead, true);
  const Index n = cfg->levelptr.back(), base = cfg->base_num, forced = heap.s;
  tide::Graph g;
  g.regions.resize(base, {budget, true, true, "content", "lh-count-affect-v1"});
  g.regions.push_back({forced, true, true, "content", "count-v1"});
  for (Index v = 0; v < n; ++v) {
    Index region;
    if (v < forced) region = base;
    else if (v < heap.point_s) region = v-forced;
    else region = (v-heap.point_s)/cfg->localnum;
    tide::Node node{region}; node.readout = "norm-fp64-v1";
    g.nodes.push_back(node); g.inputs.push_back(v); g.outputs.push_back(v);
  }
  g.compile();
  tide::Model m;
  for (Index v = 0; v < n; ++v) {
    m.nodes.push_back({at::zeros({4}, opts), at::zeros({4, 4}, opts), at::zeros({4}, opts), at::zeros({4}, opts)});
    m.input_scale.push_back(at::ones({}, opts)); m.output_scale.push_back(at::ones({}, opts));
  }
  tide::Options options; options.workers = schedule ? 3 : 1; options.packed = schedule != 0;
  std::unique_ptr<tide::Streaming> streaming;
  std::unique_ptr<tide::Frontier> frontier;
  if (schedule == 2) frontier = std::make_unique<tide::Frontier>(g, m, options);
  else streaming = std::make_unique<tide::Streaming>(g, m, options);
  tide::Continuation q; q.identity = g.identity; q.batch_size = batch_size;
  std::map<tide::Owner, Index> positions;
  Index candidate_count = 0;
  for (Index time = 0; time < ticks; ++time) {
    VPtrBatchSignals inputs(n);
    std::vector<tide::External> external;
    for (Index node = 0; node < n; ++node) {
      std::vector<Index> samples; std::vector<at::Tensor> rows;
      for (Index b = 0; b < batch_size; ++b) {
        // Whole idle ticks, independent empty samples, sparse IDs and dense ties.
        if (time % 7 == 6 || (b == 3 && time % 4 != 0) || (time > 1 && (time*13+node*7+b*3)%5 == 0)) continue;
        std::vector<double> values;
        if (time == 0) values = {3., 4., 0., 0.};
        else if (time == 1) values = {1e6, double(node%2+1), 0., 0.};
        else values = {double((time+node+b)%5-2), double((node+2*b)%4), .25, -.5};
        auto x = at::tensor(values, opts);
        rows.push_back(x); samples.push_back(b);
        auto position = positions[{b, node}]++;
        external.push_back({b, node, position, time, x}); ++candidate_count;
      }
      if (!rows.empty()) inputs[node] = std::make_shared<BatchSignals>(at::stack(rows), at::tensor(samples, at::kLong));
    }
    const auto a = heap.select(inputs), b = tensor.select(inputs);
    auto result = streaming ? streaming->run(q, external, time+1, time+1) : frontier->run(q, external, time+1, time+1);
    q = result.continuation;
    std::map<tide::Owner, at::Tensor> outputs;
    for (const auto& o : result.outputs) {
      require(o.time == time, "Tide output clock mismatch"); outputs[{o.batch, o.port}] = o.value;
    }
    size_t emitted = 0;
    for (Index node = 0; node < n; ++node) {
      require(bool(a[node]) == bool(b[node]), "original LH heap/tensor selection presence mismatch");
      if (!a[node]) continue;
      require(at::equal(a[node]->sampleids, b[node]->sampleids), "original LH heap/tensor selected IDs differ");
      require(at::equal(a[node]->x, b[node]->x), "original LH heap/tensor selected payloads differ");
      if (node < forced) require(a[node] == inputs[node] && b[node] == inputs[node], "LH forced activity changed");
      for (Index row = 0; row < a[node]->sampleids.numel(); ++row) {
        const Index sample = a[node]->sampleids[row].item<Index>();
        auto it = outputs.find({sample, node});
        require(it != outputs.end() && at::equal(it->second, a[node]->x[row]), "Tide/LH selected ID/payload mismatch");
        ++emitted;
      }
    }
    require(emitted == outputs.size(), "Tide emitted additional selections");
    for (Index region = 0; region < base; ++region) {
      const Index points = cfg->localnum+(lead ? 1 : 0);
      for (Index sample = 0; sample < batch_size; ++sample) for (Index slot = 0; slot < points; ++slot) {
        const Index node = lead && slot == 0 ? forced+region : heap.point_s+region*cfg->localnum+slot-(lead ? 1 : 0);
        const auto affected = heap.affectcountT[region][sample][slot].item<Index>();
        const auto selected = heap.selectcountT[region][sample][slot].item<Index>();
        const auto ta = tensor.affectcounts[region].defined() ? tensor.affectcounts[region][sample][slot].item<Index>() : 0;
        const auto ts = tensor.selectcounts[region].defined() ? tensor.selectcounts[region][sample][slot].item<Index>() : 0;
        require(affected == ta && selected == ts, "original LH heap/tensor histories differ");
        require(affected == count(q, sample, region, "affected", node)
                && selected == count(q, sample, region, "selected", node), "Tide/LH selector histories differ");
      }
    }
  }
  return candidate_count;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const auto args = portable_torch::parse_cli(argc, argv);
    if (args.help) { portable_torch::print_usage(std::cout, argv[0]); return 0; }
    const auto device = portable_torch::resolve_device(args);
    if (!device.is_cpu() || (args.dtype != at::kDouble && args.dtype != at::kFloat))
      throw std::invalid_argument("LH selector oracle requires CPU FP64/FP32");
    at::set_num_threads(1); at::NoGradGuard guard;
    const auto opts = at::TensorOptions().dtype(args.dtype).device(device);
    Index candidates = 0;
    for (bool lead : {false, true}) for (Index budget : {1, 2}) for (int schedule = 0; schedule < 3; ++schedule)
      candidates += check_case(opts, lead, budget, schedule);
    std::cout << "original-LH-selector: passed; 12 cases, 288 ticks, " << candidates
              << " candidate occurrences; heap/tensor and serial/packed/frontier\n";
    return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 2; }
}
