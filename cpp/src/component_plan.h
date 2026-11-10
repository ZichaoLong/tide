#pragma once
#include "greedy_plan.h"
namespace tide {
// PDG §9.5: cached static region SCCs. Runtime queues/values are never cached.
struct ComponentPlan {
  std::vector<std::vector<Index>> groups;
  std::vector<Index> owner;
  std::vector<bool> cyclic;
  std::vector<std::shared_ptr<const ClosurePlan>> closures;
  explicit ComponentPlan(const Graph&);
};
}
