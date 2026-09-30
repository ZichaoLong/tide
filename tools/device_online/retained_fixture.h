#pragma once
#include "graph_vjp_fixture.h"
#include "retained_tape.h"
namespace tide::device_online::test {
struct RetainedReference {std::vector<Result> windows;std::map<std::string,Tensor> gradients;};
Fixture retained_fixture(int shape,int variant,int64_t width);
std::vector<int64_t> retained_stops(int64_t start);
GraphCotangents retained_roots(const ReverseTape&,int window,int mode);
RetainedReference retained_reference(Fixture,int mode,at::ScalarType);
} // namespace tide::device_online::test
