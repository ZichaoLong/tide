#pragma once
#include "weighted_query_batch.h"

namespace tide::weighted::query {
std::vector<TensorFields> read_vjp(const NodeWeights&,const std::vector<ReadInput>&,const KVConfig&,const RegionalConfig&);
std::vector<TensorFields> read_vjp(const NodeWeights&,const std::vector<ReadInput>&,const RecurrentConfig&,const RegionalConfig&);
std::vector<FullResult> full_vjp(const NodeWeights&,const std::vector<FullInput>&,Index,const Options&,const KVConfig&);
std::vector<FullResult> full_vjp(const NodeWeights&,const std::vector<FullInput>&,Index,const Options&,const RecurrentConfig&);
}
