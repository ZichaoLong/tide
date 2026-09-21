#include "streaming.h"
#include <set>
#include <stdexcept>

namespace tide_bench {
Config parse(int argc, char** argv) {
  Config c; std::vector<char*> common{argv[0]}; std::set<std::string> seen;
  std::map<std::string, Index*> integers{{"--nodes", &c.nodes}, {"--batch", &c.batch},
    {"--width", &c.width}, {"--ticks", &c.ticks}, {"--active-rings", &c.rings},
    {"--workers", &c.workers}, {"--warmup", &c.warmup}, {"--repetitions", &c.repetitions}};
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i]; const auto equal = argument.find('=');
    const auto key = argument.substr(0, equal);
    if (!integers.count(key) && key != "--api" && key != "--packed" && key != "--run-id") {
      common.push_back(argv[i]); continue;
    }
    if (!seen.insert(key).second) throw std::invalid_argument("duplicate benchmark option: "+key);
    std::string value;
    if (equal != std::string::npos) value = argument.substr(equal+1);
    else {
      if (++i == argc) throw std::invalid_argument("missing value for "+key);
      value = argv[i];
    }
    if (integers.count(key)) {
      if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("nonnegative integer required: "+key);
      *integers.at(key) = std::stoll(value);
    } else if (key == "--api") c.api = value;
    else if (key == "--run-id") c.run_id = value;
    else {
      if (value != "0" && value != "1") throw std::invalid_argument("--packed requires 0 or 1");
      c.packed = value == "1";
    }
  }
  c.runtime = portable_torch::parse_cli(common.size(), common.data(), true);
  if (c.runtime.help) return c;
  if (c.nodes < 8 || c.nodes > 1000000 || c.nodes%8 || c.rings < 1 || c.rings > c.nodes/8
      || c.rings > 128 || c.batch < 1 || c.batch > 1024 || c.width < 1 || c.width > 4096
      || c.ticks < 1 || c.ticks > 100000 || c.workers < 1 || c.workers > 64
      || c.warmup > 100 || c.repetitions < 1 || c.repetitions > 1000
      || c.ticks*c.rings*c.batch > 10000000)
    throw std::invalid_argument("benchmark dimensions exceed the documented bounded workload");
  if (c.api != "cursor" && c.api != "functional") throw std::invalid_argument("--api requires cursor or functional");
  if (c.run_id.empty() || c.runtime.output_dir.empty()) throw std::invalid_argument("--run-id and a new --output-dir are required");
  return c;
}
}  // namespace tide_bench
