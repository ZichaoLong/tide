#pragma once
#include "weighted_column_layout.h"
namespace tide::weighted::columns {
struct QueryInputs {Tensor u,mass,weights,age,reverse;TensorFields raw;};
Tensor rotate_queries(const Tensor&,const Tensor&,const std::vector<double>&);
QueryInputs query_inputs(const KVWindow&,const Layout&,const TensorFields&,const std::vector<Index>&,
                         bool before,Index horizon,bool summary=false);
}
