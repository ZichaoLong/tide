#pragma once
#include "tide/types.h"
namespace tide::weighted {
// KV §1.2: common (normal,base) or private (values,summaries), independent roots.
std::vector<std::pair<Tensor,Tensor>> prepare_kv_rows(bool common,
    const std::vector<Tensor>& weights,const std::vector<std::pair<Tensor,Tensor>>& rows,double epsilon=0);
}
