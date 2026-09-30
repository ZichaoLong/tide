#pragma once
#include "tide/frontier.h"

namespace tide {
class ClosurePlan {
 public:
  explicit ClosurePlan(const Graph&);
  std::map<Index, std::vector<Frame>> ready(const Graph&, const Fibers&, Index stop,
                                           Index limit, Index& relaxations) const;
 private:
  std::vector<std::vector<std::pair<Index, Index>>> outgoing_;  // target region, delay
};
}  // namespace tide
