#include "tide/full_rows.h"
#include "tide/isolated_linear.h"
#include "tide/lh_full.h"
#include "tide/operator_work.h"

namespace tide {
std::vector<Tensor> full_fresh_rows(const NodeWeights& w, const std::vector<FullInput>& inputs, bool identity) {
  std::vector<Tensor> rows;
  for (const auto& in : inputs) rows.push_back(identity || w.full_kind == "identity"
                                            ? in.content.value : in.comparison->value);
  if (identity || w.full_kind == "identity") return rows;
  if (is_lh_full(w.full_kind)) {
    for (auto& row : rows) row = lh_full_fresh(w, row);
    return rows;
  }
  const auto d = w.bias.numel();
  if (w.full_kind == "swiglu") {
    work::linear(work::FullCalls, rows.size(), d, 2*d);
    work::linear(work::FullCalls, rows.size(), d, 2*d);
    work::linear(work::FullCalls, rows.size(), 2*d, d);
    auto gate = isolated_linear(rows, w.extra.at("ffn_gate").t());
    auto up = isolated_linear(rows, w.extra.at("ffn_up").t());
    for (size_t i = 0; i < rows.size(); ++i) gate[i] = at::silu(gate[i])*up[i];
    rows = isolated_linear(gate, w.extra.at("ffn_down").t());
  } else {
    work::linear(work::FullCalls, rows.size(), d, d);
    rows = isolated_linear(rows, w.weight.t());
    for (auto& row : rows) row = at::tanh(row+w.bias);
  }
  for (size_t i = 0; i < rows.size(); ++i) rows[i] = inputs[i].content.value+rows[i];
  return rows;
}
}  // namespace tide
