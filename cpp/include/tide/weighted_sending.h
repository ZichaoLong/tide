#pragma once
// Pinned allocation §4.2–4.4 / §6.3; docs/weighted-sending.md owns VJP boundaries.
#include "tide/types.h"
#include <string>
#include <vector>

namespace tide::weighted {
Tensor attention_shares(const Tensor& scores,const Tensor& base,const Tensor& temperature);
Tensor attention_send(const Tensor& scores,const Tensor& caps,const Tensor& received,
                      const std::string& mode,const std::string& base,const Tensor& temperature);
Tensor proportional(const Tensor& caps,Index k);
Tensor total_zero(const Tensor& scores,const Tensor& caps,const Tensor& total,const std::vector<Index>& nodes);
Tensor hard_selection(const Tensor& scores,const Tensor& caps,Index k,const std::vector<Index>& nodes);
}
