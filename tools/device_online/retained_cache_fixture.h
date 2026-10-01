#pragma once
#include "retained_fixture.h"

namespace tide::device_online::test {
// profile0:event;1..5:fiber pools;6:mixed event/fiber. Ordinary public inputs.
Fixture retained_cache_fixture(int shape,int variant,Index width,int profile,bool clock=false);
void retained_cache_roots(GraphCotangents&,const ReverseTape&,int window,int mode);
void compare_retained_cache(const GraphVjp&,const ReverseTape&,const RetainedReference&,const Fixture&,bool half);
void poison_live_cache(ReverseTape&);
std::vector<Tensor> cache_gradient_values(const GraphVjp&);
} // namespace tide::device_online::test
