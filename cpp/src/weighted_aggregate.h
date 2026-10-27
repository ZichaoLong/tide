#pragma once
#include "tide/aggregate.h"
#include "tide/source_class.h"

namespace tide::weighted {
std::vector<AggregateResult> aggregate_batch(const NodeWeights&,const std::vector<AggregateInput>&,
    const SourceClasses&,double alpha,double epsilon);
std::vector<AggregateResult> aggregate_batch_vjp(const NodeWeights&,const std::vector<AggregateInput>&,
    const SourceClasses&,double alpha,double epsilon);
}
