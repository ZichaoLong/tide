#include "tide/ops.h"
#include <algorithm>
#include <stdexcept>
#include <ATen/core/grad_mode.h>

namespace tide {
namespace {
void finite_inputs(const std::vector<External>& values){
  at::NoGradGuard guard;
  constexpr int64_t budget=8*1024*1024;
  std::map<std::tuple<int,int,int>,std::vector<Tensor>> groups;
  for(const auto& x:values){const auto& t=x.value;
    groups[{int(t.device().type()),t.device().index(),int(t.scalar_type())}].push_back(t);}
  auto check=[](const Tensor& t){if(!at::isfinite(t).all().item<bool>())throw std::invalid_argument("nonfinite tensor");};
  for(const auto& [key,tensors]:groups){
    std::vector<Tensor> rows;int64_t bytes=0;
    for(const auto& t:tensors){const auto count=t.numel()*t.element_size();
      if(!rows.empty()&&bytes+count>budget){check(at::cat(rows));rows.clear();bytes=0;}
      if(count>budget)check(t);else{rows.push_back(t.reshape({-1}));bytes+=count;}
    }
    if(!rows.empty())check(at::cat(rows));
  }
}
}
ValidatedInput validate_external(const Graph& g, const Model& m, const Continuation& q,
                                 const std::vector<External>& external, Index stop, Index seal) {
  // q's graph identity and imported state were already validated. Only touched
  // ports enter this transaction; no copy or scan of the accumulated ledger.
  if (q.cut < 0 || stop < q.cut || seal < stop) throw std::invalid_argument("window is unsealed or precedes cut");
  ValidatedInput result;
  auto sorted = external;
  std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
    return std::tie(a.batch, a.port, a.position) < std::tie(b.batch, b.port, b.position);
  });
  for (const auto& x : sorted) {
    if (x.batch < 0 || x.batch >= q.batch_size || x.port < 0 || x.port >= static_cast<Index>(g.inputs.size())
        || x.position < 0 || x.time < q.cut || x.time >= stop)
      throw std::invalid_argument("invalid external coordinate");
    const Owner owner{x.batch, x.port};
    auto changed = result.ledger_updates.find(owner);
    auto stored = q.ledger.find(owner);
    auto last = changed != result.ledger_updates.end() ? changed->second : stored != q.ledger.end() ? stored->second : Owner{-1, -1};
    // Avoid overflowing last.position+1 when a ledger is exhausted.
    if (x.position == 0 ? last.first != -1 : x.position - 1 != last.first)
      throw std::invalid_argument("noncontiguous/nonmonotonic port history");
    if (x.time <= last.second) throw std::invalid_argument("noncontiguous/nonmonotonic port history");
    const auto& value = x.value;
    const auto& ref = m.nodes[g.inputs[x.port]].bias;
    if (!value.defined() || value.device() != ref.device() || value.scalar_type() != ref.scalar_type()
        || value.sizes() != at::IntArrayRef({m.width()})) throw std::invalid_argument("incompatible tensor dtype/device/shape");
    result.ledger_updates[owner] = {x.position, x.time};
    result.atoms.push_back({x.batch, g.inputs[x.port], x.time, 0, x.port, x.position, value});
  }
  finite_inputs(sorted);
  return result;
}
}  // namespace tide
