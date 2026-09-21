// Small versioned, dtype/shape-explicit exchange with the independent Python oracle.
#include "lh_iocortex_fixture.h"
#include <filesystem>
#include <fstream>

namespace lh_iocortex {
namespace {
using Json = nlohmann::json;
Json tensor(const at::Tensor& x) {
  auto flat = x.to(at::kDouble).contiguous().reshape({-1});
  const auto* data = flat.data_ptr<double>();
  std::vector<double> values; if (flat.numel()) values.assign(data, data+flat.numel());
  return {{"shape", x.sizes().vec()}, {"values", values}};
}
Json cache(const Cache& values) {
  Json result = Json::object(); for (const auto& [key, value] : values) result[key] = tensor(value); return result;
}
Json graph(const tide::Graph& g) {
  Json result{{"inputs", g.inputs}, {"outputs", g.outputs}, {"nodes", Json::array()},
              {"edges", Json::array()}, {"regions", Json::array()}};
  for (const auto& n : g.nodes)
    result["nodes"].push_back({{"region", n.region}, {"clear", n.clear}, {"memory", n.memory}, {"full", n.full},
      {"query_heads", n.query_heads}, {"kv_heads", n.kv_heads}, {"emission", n.emission}, {"readout", n.readout}});
  for (const auto& e : g.edges) result["edges"].push_back({{"source", e.source}, {"target", e.target}, {"delay", e.delay}});
  for (const auto& r : g.regions)
    result["regions"].push_back({{"budget", r.budget}, {"observe_all", r.observe_all}, {"count_priority", r.count_priority},
      {"read_mode", r.read_mode}, {"selector", r.selector}});
  result["layout"] = {{"edge_source", g.layout->edge_source}, {"edge_target", g.layout->edge_target},
                       {"input", g.layout->input}, {"output", g.layout->output}};
  return result;
}
Json model(const tide::Model& m) {
  Json result; result["nodes"] = Json::array();
  for (const auto& w : m.nodes)
    result["nodes"].push_back({{"decay", tensor(w.decay)}, {"weight", tensor(w.weight)},
      {"bias", tensor(w.bias)}, {"read", tensor(w.read)}, {"extra", cache(w.extra)}});
  const std::map<std::string, const std::vector<at::Tensor>*> scales{{"input_scale", &m.input_scale},
    {"agg_scale", &m.agg_scale}, {"edge_scale", &m.edge_scale}, {"output_scale", &m.output_scale}};
  for (const auto& [name, values] : scales) {
    result[name] = Json::array(); for (const auto& value : *values) result[name].push_back(tensor(value));
  }
  return result;
}
}  // namespace

void write_fixture(const std::string& path, const Fixture& f, const std::vector<tide::External>& inputs,
                   const ObservedSelector& is, const ObservedSelector& os,
                   const VPtrBatchPtrBaseHidden& ih, const VPtrBatchPtrBaseHidden& oh,
                   const PtrBatchPtrBaseHidden& ph, const std::vector<at::Tensor>& logits) {
  if (path.empty()) return;
  const Index cut = is.ticks.size();
  Json record{{"schema", "lh-iocortex-fixture-v1"}, {"dtype", f.opts.dtype().toScalarType() == at::kDouble ? "float64" : "float32"},
    {"batch_size", Fixture::batch}, {"width", Fixture::width}, {"layers", f.policy.layers()}, {"cut", cut},
    {"pool", f.policy.pool}, {"clear", f.policy.clear}, {"scenario", ph ? "tokens" : "ragged"},
    {"body", graph(f.body)}, {"model", model(f.model)}, {"readout", graph(f.readout)}, {"read_model", model(f.read_model)}};
  for (const auto& field : {"inputs", "events", "outputs", "hidden", "counters", "pending", "read_hidden", "logits"}) record[field] = Json::array();
  for (const auto& x : inputs) record["inputs"].push_back({x.batch, x.port, x.position, x.time, tensor(x.value)});
  for (Index t = 0; t < cut; ++t) for (Index c = 0; c < 2; ++c) {
    const auto& tick = (c ? os : is).ticks.at(t);
    for (const auto& [owner, value] : tick.proposals) {
      at::Tensor full; const auto& selected = tick.selected.at(owner.second);
      if (selected) for (Index row = 0; row < selected->sampleids.numel(); ++row)
        if (selected->sampleids[row].item<Index>() == owner.first) full = selected->x[row];
      record["events"].push_back({t, owner.first, c*f.n+owner.second, tensor(value), full.defined(), full.defined() ? tensor(full) : Json()});
      if (c && owner.second == 0 && full.defined()) record["outputs"].push_back({owner.first, t, 0, tensor(full)});
    }
  }
  for (Index c = 0; c < 2; ++c) {
    const auto& s = (c ? os : is).original;
    for (Index v = 0; v < f.n; ++v) for (Index b = 0; b < Fixture::batch; ++b)
      record["hidden"].push_back({b, c*f.n+v, cache(original_cache((c ? oh : ih).at(v), b, f.opts))});
    for (Index r = 0; r < f.cfg->base_num; ++r) for (Index b = 0; b < Fixture::batch; ++b)
      for (Index slot = 0; slot < f.cfg->localnum+(f.policy.lead ? 1 : 0); ++slot) {
        const auto v = f.policy.lead && slot == 0 ? s.s+r : s.point_s+r*f.cfg->localnum+slot-(f.policy.lead ? 1 : 0);
        require(!s.select_by_tensor, "fixture export expects original heap counter storage");
        record["counters"].push_back({b, c*(f.cfg->base_num+1)+r, c*f.n+v,
          s.selectcountT[r][b][slot].item<Index>(), s.affectcountT[r][b][slot].item<Index>()});
      }
  }
  for (Index block = 0; block < 4; ++block) {
    const auto values = f.blocks[block]->BaseEmitToEdges(Fixture::batch, (block%2 ? os : is).ticks.back().selected);
    for (Index edge = 0; edge < static_cast<Index>(values.size()); ++edge) if (values[edge])
      for (Index row = 0; row < values[edge]->sampleids.numel(); ++row) {
        auto e = f.edge_offsets[block]+edge;
        record["pending"].push_back({values[edge]->sampleids[row].item<Index>(), f.body.edges[e].target,
          cut, 1, e, cut-1, tensor(values[edge]->x[row])});
      }
  }
  if (ph) for (Index b = 0; b < Fixture::batch; ++b)
    record["read_hidden"].push_back({b, 0, cache(original_cache(ph, b, f.opts))});
  for (const auto& value : logits) record["logits"].push_back(tensor(value));
  require(!std::filesystem::exists(path), "refusing to overwrite original IOCortex fixture");
  std::ofstream stream; stream.exceptions(std::ios::failbit | std::ios::badbit); stream.open(path);
  stream << record.dump() << '\n'; stream.close();
}
}  // namespace lh_iocortex
