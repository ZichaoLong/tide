#pragma once
#include "content_fixture.h"

namespace tide::device_online::test {
// CPU-only oracle adapters. Forward values obey the declared half operator
// boundaries; autograd leaves/adjoints stay FP32 or FP64. No device tape or
// candidate schedule is an input to this reference.
void fixture_dtype(Fixture&,at::ScalarType);
void configure_half_reference(Fixture&);
void round_half_transport(Result&);
void check_half_reference_norm();
} // namespace tide::device_online::test
