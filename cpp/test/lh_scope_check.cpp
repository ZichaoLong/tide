#include "lh_iocortex_scope.h"
#include <iostream>
#include <map>
#include <set>

namespace {
void require(bool condition) {
  if (!condition) throw std::runtime_error("IOCortex scope contract failed");
}
lh_iocortex::OracleScope parse(std::vector<std::string> args) {
  std::vector<char*> pointers;
  for (auto& arg : args) pointers.push_back(arg.data());
  auto result = lh_iocortex::oracle_scope(pointers.size(), pointers.data());
  // Only the name is used after the local argument storage expires.
  result.arguments.clear(); return result;
}
}
int main() {
  try {
    using lh_iocortex::OracleScope;
    OracleScope full, smoke; smoke.name = "smoke";
    std::map<std::string, int> all, small; std::set<int> modes, schedules, clears, leads;
    int full_count = 0, smoke_count = 0, unavailable = 0;
    for (const auto& pool : {"add", "sum", "mean", "linear", "active-softmax", "all-softmax"})
      for (bool clear : {false, true}) for (bool lead : {false, true})
        for (int mode = 0; mode < 3; ++mode) for (int schedule = 0; schedule < 3; ++schedule) {
          if (std::string(pool) == "add" && mode == 2) continue;
          require(full.includes(pool, clear, lead, mode, schedule));
          ++all[pool]; ++full_count;
          if (std::string(pool) == "active-softmax" && mode) ++unavailable;
          if (smoke.includes(pool, clear, lead, mode, schedule)) {
            ++small[pool]; ++smoke_count; modes.insert(mode); schedules.insert(schedule);
            clears.insert(clear); leads.insert(lead);
            require(std::string(pool) != "active-softmax" || mode == 0);
          }
        }
    require(full_count == 204 && unavailable == 24 && smoke_count == 6);
    for (const auto& [pool, count] : all) require(count == (pool == "add" ? 24 : 36) && small[pool] == 1);
    require(modes.size() == 3 && schedules.size() == 3 && clears.size() == 2 && leads.size() == 2);
    require(parse({"check"}).name == "full");
    require(parse({"check", "--scope=smoke"}).name == "smoke");
    require(parse({"check", "--device", "cpu", "--scope", "full"}).name == "full");
    for (const auto& args : std::vector<std::vector<std::string>>{
          {"check", "--scope"}, {"check", "--scope="}, {"check", "--scope", "quick"},
          {"check", "--scope=smoke", "--scope", "full"}}) {
      bool rejected = false;
      try { parse(args); } catch (const std::invalid_argument&) { rejected = true; }
      require(rejected);
    }
    char name[] = "check", scope[] = "--scope=smoke", help[] = "--help";
    char* args[] = {name, scope, help}; auto filtered = lh_iocortex::oracle_scope(3, args);
    require(filtered.arguments.size() == 2 && filtered.arguments[0] == name && filtered.arguments[1] == help);
    std::cout << "lh-iocortex-scope: full=204, assertions-on-fp64=180, smoke=6; passed\n";
    return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
