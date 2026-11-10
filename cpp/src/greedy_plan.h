#pragma once
#include "tide/frontier.h"

namespace tide {
class ClosurePlan {
 public:
  explicit ClosurePlan(const Graph&);
  ClosurePlan(const Graph&,const std::vector<Index>& regions);
  ClosurePlan(const Graph&,const std::vector<Index>& regions,const std::vector<Index>& edges);
  std::map<Index, std::vector<Frame>> ready(const Graph&, const Fibers&, Index stop,
                                           Index limit, Index& relaxations) const;
 private:
  std::vector<std::vector<std::pair<Index, Index>>> outgoing_;  // target region, delay
  std::map<Index,Index> local_;
};
}  // namespace tide
