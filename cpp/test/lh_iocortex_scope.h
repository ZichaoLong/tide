#pragma once
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace lh_iocortex {
// Test-only scope: the default full matrix and fixture inventory stay unchanged.
struct OracleScope {
  std::string name = "full";
  std::vector<char*> arguments;
  bool includes(const std::string& pool, bool clear, bool lead, int mode, int schedule) const {
    if (name == "full") return true;
    using Case = std::tuple<std::string, bool, bool, int, int>;
    const Case actual{pool, clear, lead, mode, schedule};
    for (const auto& c : std::vector<Case>{{"add", false, false, 0, 0}, {"sum", true, true, 2, 2},
          {"mean", false, true, 1, 1}, {"linear", true, false, 1, 2},
          {"active-softmax", false, true, 0, 2}, {"all-softmax", true, false, 2, 1}})
      if (c == actual) return true;
    return false;
  }
};

inline OracleScope oracle_scope(int argc, char** argv) {
  OracleScope result; bool seen = false;
  for (int i = 0; i < argc; ++i) {
    const std::string arg = argv[i];
    if (i && (arg == "--scope" || arg.rfind("--scope=", 0) == 0)) {
      if (seen) throw std::invalid_argument("duplicate --scope");
      seen = true;
      if (arg == "--scope") {
        if (++i == argc) throw std::invalid_argument("--scope requires full or smoke");
        result.name = argv[i];
      } else result.name = arg.substr(8);
      if (result.name != "full" && result.name != "smoke")
        throw std::invalid_argument("--scope must be full or smoke");
    } else result.arguments.push_back(argv[i]);
  }
  return result;
}
}  // namespace lh_iocortex
