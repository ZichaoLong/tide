#pragma once
#include "placement.h"
#include <tide/region.h>

namespace accelerator_scale {
// Separate candidates: ranking and event-time/order decisions. Host code still
// owns Tensor handles, commits histories and dispatches the selected local work.
struct RegionTask {
  Owner owner;
  const std::vector<size_t>* ids;
  const History* old;
  Selection selection;
};
Tensor rank_candidates(const Tensor& scores, const Tensor& selected, const Tensor& affected);
void tensor_select(const Graph&, const Model&, const Placement&, std::vector<Event>&,
                   const std::vector<RegionTask*>&);

class TensorEventQueue {
 public:
  explicit TensorEventQueue(at::Device);
  void push(const Atom&);
  bool pop(Index stop, std::vector<Atom>&);
  void export_pending(Continuation&) const;
 private:
  at::Device device_;
  Tensor keys_;
  std::vector<Atom> pending_, staged_;
  void flush();
};
}  // namespace accelerator_scale
