#pragma once
#include "weighted_detail.h"
#include "tide/weighted_convolution.h"

namespace tide::weighted::recurrent {
using namespace detail;
inline ConvCache cache(const State& s){return {s.slots.at("conv_times"),s.slots.at("conv_values")};}
std::array<Tensor,3> effective(const State&,Index,const RecurrentConfig&);
Tensor finish(const NodeWeights&,const Tensor& u,const Tensor& mix,const RecurrentConfig&);
Tensor basic_update(const NodeWeights&,const Tensor& u,const Tensor& receive,const Tensor& H,const RecurrentConfig&);
Tensor basic_read(const NodeWeights&,const Tensor& u,const Tensor& H,const RecurrentConfig&);
Tensor basic_mix(const NodeWeights&,const Tensor& u,const Tensor& H,const RecurrentConfig&);
std::pair<Tensor,Tensor> gdn_update(const NodeWeights&,const State&,Index,const Tensor&,const Tensor&,const Tensor&,const RecurrentConfig&);
Tensor gdn_read(const NodeWeights&,const State&,Index,const Tensor&,const RecurrentConfig&);
Tensor gdn_mix(const NodeWeights&,const State&,Index,const Tensor&,const RecurrentConfig&);
std::pair<Tensor,Tensor> ssd_update(const NodeWeights&,const State&,Index,const Tensor&,const Tensor&,const Tensor&,const RecurrentConfig&);
Tensor ssd_read(const NodeWeights&,const State&,Index,const Tensor&,const RecurrentConfig&);
Tensor ssd_mix(const NodeWeights&,const State&,Index,const Tensor&,const RecurrentConfig&);
void validate_basic(const NodeWeights&,const RecurrentConfig&);
void validate_gdn(const NodeWeights&,const RecurrentConfig&);
void validate_ssd(const NodeWeights&,const RecurrentConfig&);
}
namespace tide::weighted {
void validate_content_weights(const NodeWeights&,const KVConfig&);
void validate_content_weights(const NodeWeights&,const RecurrentConfig&);
}
