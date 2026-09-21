#pragma once
#include "tide/types.h"

namespace tide {
using FiberViews = std::vector<const std::vector<Atom>*>;
// Nonempty segments for one node/program. Fibers retain source tags and positions.
// Views are borrowed for the duration of the kernel call, never stored in State.
struct PackedSequence {
  Tensor contents;  // [sum(lengths), width]
  std::vector<Index> offsets{0};
  std::vector<Owner> owners;  // sample, node
  std::vector<Index> times;
  FiberViews fibers;
  void validate() const;
};
struct PackedStates {
  std::vector<State> states;  // same flattened order as contents
  Index calls = 0, max_batch = 0, max_length = 0, score_elements = 0;
};
}  // namespace tide
