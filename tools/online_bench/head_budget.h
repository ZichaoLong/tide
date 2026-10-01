#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace tide_flow {
struct HeadBudget {int64_t rows,fixed_bytes,row_bytes,reserved_bytes,budget,operator_allowance_bytes;};
// Tensor envelope for the consumer head, including one accumulated head
// gradient, temporary partial and a calibrated operator allowance (the local
// CANN matmul path needs 16MiB even for one row; reserve twice that). This is an
// estimate, not a guarantee for other vendor versions. Graph tapes/optimizer
// have separate owners. No logical output/loss/update boundary is split here.
inline HeadBudget head_budget(int64_t capacity,int64_t width,int64_t vocab,int64_t payload,
    bool backward,int64_t budget,bool aggressive) {
  if(capacity<1||width<1||vocab<1||(payload!=2&&payload!=4)||budget<1)
    throw std::invalid_argument("invalid head workspace geometry/budget");
  constexpr int64_t operators=32LL*1024*1024;
  const long double fixed=operators+4096.L+8.L*capacity+(backward?
    4.L*capacity*width+4.L*(3+(payload==2))*vocab*width:0.L);
  const long double row=(backward?32.L:16.L)*vocab+(payload+(backward?12.L:4.L))*width+160.L;
  const auto usable=budget-budget/(aggressive?10:4);
  if(fixed+row>usable)throw std::invalid_argument("one output head row exceeds head-workspace-bytes");
  const auto rows=static_cast<int64_t>(std::min<long double>(capacity,(usable-fixed)/row));
  return {rows,static_cast<int64_t>(fixed),static_cast<int64_t>(row),
    static_cast<int64_t>(fixed+rows*row),budget,operators};
}
} // namespace tide_flow
