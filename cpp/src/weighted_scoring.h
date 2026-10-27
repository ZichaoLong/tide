#pragma once
// Allocation §3: concrete independently parameterized finite-dimensional maps.
#include "weighted_detail.h"

namespace tide::weighted::detail {
using Groups=std::array<Tensor,3>; // normalized current, two memories, three logs
Tensor encode_group(const NodeWeights&,const std::string&,const Tensor&,double epsilon);
Tensor local_score(const NodeWeights&,const RegionalConfig&,const Tensor&,const Groups&,double epsilon);
Tensor region_evidence(const NodeWeights&,const RegionalConfig&,const Tensor&,const Groups&,double epsilon);
Tensor memory_key(const NodeWeights&,const RegionalConfig&,const Groups&,double epsilon);
void validate_describe(const NodeWeights&,const RegionalConfig&,Index width,Index summary_width);
Tensor region_scores(const RegionWeights&,const RegionalConfig&,const std::map<std::string,Tensor>&);
void validate_query(const RegionWeights&,const RegionalConfig&);
}
