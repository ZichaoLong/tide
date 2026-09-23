#include "fiber_pool.h"
#include "tide/operator_profile.h"
#include <stdexcept>

namespace tide {
Tensor fiber_pool_csr(const NodeWeights& w, const std::string& kind, const std::vector<Index>& slots,
                      const std::vector<Index>& offsets, const std::vector<Tensor>& event_rows) {
  op_profile::Scope profile(op_profile::Pooling);
  if (event_rows.empty() || offsets.size() < 2 || offsets.front() != 0
      || offsets.back() != static_cast<Index>(slots.size()))
    throw std::invalid_argument("invalid fiber pooling CSR offsets");
  std::vector<Index> lengths;
  for (size_t i = 1; i < offsets.size(); ++i) {
    if (offsets[i] <= offsets[i-1]) throw std::invalid_argument("empty fiber pooling CSR row");
    lengths.push_back(offsets[i]-offsets[i-1]);
  }
  auto rows = event_rows.size() == 1 ? event_rows.front() : at::cat(event_rows);
  if (rows.dim() != 2 || rows.size(0) != offsets.back())
    throw std::invalid_argument("fiber pooling CSR row count mismatch");
  const auto indices = rows.options().dtype(at::kLong);
  Tensor coefficients;
  if (kind == "sum" || kind == "mean") coefficients = at::ones({rows.size(0)}, rows.options());
  else {
    if (!fiber_pool_learned(kind)) throw std::invalid_argument("unknown fiber pooling kind");
    auto weight = w.extra.at("fiber_pool");
    if (kind == "all-softmax") weight = at::softmax(weight, 0);
    coefficients = weight.index_select(0, at::tensor(slots, indices));
    if (kind == "active-softmax") {
      // Normalize on each active domain directly, including extreme logits.
      // Full-domain softmax followed by renormalization can underflow to 0/0.
      std::vector<Tensor> local;
      for (size_t i = 1; i < offsets.size(); ++i)
        local.push_back(at::softmax(coefficients.slice(0, offsets[i-1], offsets[i]), 0));
      coefficients = at::cat(local);
    }
  }
  auto matrix = at::sparse_csr_tensor(at::tensor(offsets, indices), at::arange(rows.size(0), indices),
                                    coefficients, {static_cast<Index>(lengths.size()), rows.size(0)},
                                    rows.options().layout(at::kSparseCsr));
  auto pooled = at::matmul(matrix, rows);
  if (kind == "mean") pooled = pooled/at::tensor(lengths, indices).to(rows.scalar_type()).unsqueeze(1);
  return pooled;
}
} // namespace tide
