#pragma once
#include "tide/kernel.h"
namespace tide::weighted {
bool bind_kv_hidden_sequence(const NodeWeights&,const std::vector<State>&,
    const PackedSequence&,std::vector<State>&,std::vector<State>&);
}
