#pragma once
#include "weighted_packed.h"
namespace tide::weighted::packed {
// Actual positive-write rows; common operands retain every user's adjoint.
std::pair<Tensor,Tensor> kv_projections(const NodeWeights&,const Writes&,const PackedSequence&,const KVConfig&);
}
