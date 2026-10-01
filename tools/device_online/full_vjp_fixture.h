#pragma once
#include "full_vjp.h"
#include <vector>

namespace tide::device_online::test {
struct FullFixture {
  FullTape tape;
  at::Tensor gradient,connected;
};
FullFixture full_fixture(int64_t width,int mode,bool tanh);
// Ordinary CPU autograd, independent of device chunk/owner planning. Returns
// content rows, comparison rows, weight owners and bias owners in that order.
std::vector<at::Tensor> full_reference(const FullFixture&,at::ScalarType);
void full_same(const at::Tensor&,const at::Tensor& connected,const at::Tensor& expected,const char* field);
void full_same_precision(const at::Tensor&,const at::Tensor& connected,const at::Tensor& expected,const char* field,bool half);
void full_compare(const FullFixture&,const FullVjp&,at::ScalarType);
} // namespace tide::device_online::test
