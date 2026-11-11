#pragma once
// Typed recurrent windows share messages/regions, not KV's private shape.
#include "weighted_column_layout.h"
namespace tide::weighted::columns::rec {
struct Features {Tensor query,key,value,gate,delta;};
struct Coefficients {Tensor a,b;bool matrix;};
struct Snapshot {Layout layout;TensorFields fields;std::vector<Index> times;Layout cache_layout;Tensor cache_values;};
std::pair<Layout,Tensor> cache_bank(const NodeWeights&,const RecurrentConfig&,const Layout&,
                                  const Tensor&,const Tensor&,const RecurrentState&);
Features features(const NodeWeights&,const RecurrentConfig&,const Layout&,const Tensor&,const Layout&,const Tensor&);
Coefficients coefficients(const NodeWeights&,const RecurrentConfig&,const Layout&,const Tensor&,
                          const Tensor&,const Tensor&,const Tensor&,const Layout&,const Tensor&);
Snapshot snapshot(const RecurrentWindow&,const std::vector<Index>&,const std::string&,const RecurrentState&);
std::array<Tensor,3> effective(const Snapshot&,const RecurrentConfig&);
}
