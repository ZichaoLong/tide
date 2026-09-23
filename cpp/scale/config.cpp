#include "scale.h"
#include <fstream>
#include <set>
#include <stdexcept>

namespace pdg_scale {
Config parse(int argc, char** argv) {
  Config c; std::vector<char*> common{argv[0]}; std::set<std::string> seen;
  std::map<std::string, Index*> ints{{"--width", &c.width}, {"--batch", &c.batch}, {"--steps", &c.steps},
    {"--warmup", &c.warmup}, {"--workers", &c.workers}, {"--threads", &c.threads}, {"--vocab", &c.vocab},
    {"--head-workers", &c.head_workers}};
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (!ints.count(key) && key != "--topology" && key != "--run-id" && key != "--emission" && key != "--attention-packing"
        && key != "--packed" && key != "--grad" && key != "--check" && key != "--profile"
        && key != "--work-count" && key != "--operator-profile" && key != "--parallel-regions" && key != "--compact-events") { common.push_back(argv[i]); continue; }
    if (!seen.insert(key).second || ++i == argc) throw std::invalid_argument("duplicate/missing option: "+key);
    std::string value = argv[i];
    if (ints.count(key)) {
      if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("integer required: "+key);
      *ints.at(key) = std::stoll(value);
    } else if (key == "--topology") c.topology = value;
    else if (key == "--run-id") c.run_id = value;
    else if (key == "--emission") c.emission = value;
    else if (key == "--attention-packing") c.attention_packing = value;
    else {
      if (value != "0" && value != "1") throw std::invalid_argument("boolean requires 0 or 1: "+key);
      if (key == "--packed") c.packed = value == "1";
      if (key == "--grad") c.grad = value == "1";
      if (key == "--check") c.check = value == "1";
      if (key == "--work-count") c.work_count = value == "1";
      if (key == "--operator-profile") c.operator_profile = value == "1";
      if (key == "--profile") c.profile = value == "1";
      if (key == "--parallel-regions") c.parallel_regions = value == "1";
      if (key == "--compact-events") c.compact_events = value == "1";
    }
  }
  c.runtime = portable_torch::parse_cli(common.size(), common.data(), true);
  if (c.runtime.help) return c;
  if (c.width < 4 || c.width > 4096 || c.width%4 || c.batch < 1 || c.batch > 1024
      || c.steps < 1 || c.steps > 1000 || c.warmup >= c.steps || c.workers < 1 || c.workers > 160
      || c.threads < 1 || c.threads > 160 || c.workers*c.threads > 160 || c.vocab < 2 || c.vocab > 100000
      || c.head_workers < 1 || c.head_workers > 160 || c.head_workers*c.threads > 160
      || (c.emission != "row" && c.emission != "slot") || c.topology.empty()
      || (c.attention_packing != "exact" && c.attention_packing != "single")
      || c.run_id.empty() || c.runtime.output_dir.empty()) throw std::invalid_argument("invalid bounded PDG scale configuration");
  if (c.operator_profile && (c.grad || !c.packed || c.emission != "row"))
    throw std::invalid_argument("operator profiling supports packed inference row Emit only");
  if (c.work_count && (c.grad || c.emission != "row"))
    throw std::invalid_argument("work accounting supports inference row Emit only");
  if (c.check && (c.width > 64 || c.batch > 8 || c.steps > 12))
    throw std::invalid_argument("full traced check is restricted to small shapes");
  return c;
}
Topology read_topology(const std::string& path) {
  std::ifstream in(path); std::string magic; Topology t{}; Index edges = 0;
  in >> magic >> t.nodes >> t.local >> t.points >> t.forced >> t.budget >> t.layers >> edges;
  if (!in || magic != "TIDE_PDG_SCALE_1" || t.nodes < 2 || t.nodes > 200000 || t.local < 1
      || t.forced < 1 || t.points <= t.forced || t.points >= t.nodes || t.budget < 1 || t.budget > t.local
      || t.nodes-t.points != (t.points-t.forced)*t.local || t.layers < 1 || t.layers > 8
      || edges < 1 || edges > 4000000) throw std::invalid_argument("invalid PDG topology header");
  for (Index i = 0; i < edges; ++i) {
    Index source, target; in >> source >> target;
    if (!in || source < 0 || target < 0 || source >= 2*t.nodes || target >= 2*t.nodes)
      throw std::invalid_argument("invalid PDG topology edge");
    t.edges.push_back({source, target, 1});
  }
  if (in >> magic) throw std::invalid_argument("trailing topology data");
  return t;
}
} // namespace pdg_scale
