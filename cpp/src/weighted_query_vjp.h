#pragma once
#include "weighted_query_batch.h"

namespace tide::weighted::query {
std::vector<TensorFields> read_vjp(const NodeWeights&,const std::vector<ReadInput>&,const KVConfig&,const RegionalConfig&,bool dense=false);
std::vector<TensorFields> read_vjp(const NodeWeights&,const std::vector<ReadInput>&,const RecurrentConfig&,const RegionalConfig&,bool dense=false);
std::vector<FullResult> full_vjp(const NodeWeights&,const std::vector<FullInput>&,Index,const Options&,const KVConfig&,bool dense=false);
std::vector<FullResult> full_vjp(const NodeWeights&,const std::vector<FullInput>&,Index,const Options&,const RecurrentConfig&,bool dense=false);
}
