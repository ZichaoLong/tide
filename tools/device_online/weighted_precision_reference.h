#pragma once
#include "tide/types.h"
namespace tide::device_online::weighted_precision_check {
// Independent scalar CPU schedule uses core textbook functions in FP32, with
// explicit raw/payload storage boundaries and identity-cast FP32 VJPs. It never
// calls resident numerical programs, packing, decode or candidate scheduling.
Model reference_model(const Model& quantized_parameters);
std::vector<External> reference_inputs(std::vector<External>);
Continuation reference_continuation(Continuation);
// Full public training field is the FP32 value before payload storage.
Tensor unrounded_full(const Model&,const Event&);
void validate_storage(const Result&,at::ScalarType);
}
