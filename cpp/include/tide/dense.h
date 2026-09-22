#pragma once
#include "tide/types.h"
#include "tide/pool.h"

namespace tide {
// Independent output-column blocks, with ordinary differentiable slice/cat.
// This pool is used between graph advances; it does not change global BLAS state.
class DenseLinear {
 public:
  explicit DenseLinear(Index workers) : workers_(workers), pool_(workers) {}
  Tensor run(const Tensor& input, const Tensor& weight, const Tensor& bias = {});
 private:
  Index workers_;
  NodePool pool_;
  std::mutex mutex_;
};
} // namespace tide
