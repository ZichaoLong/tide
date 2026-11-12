#pragma once
#include "device_program.h"
#include "weighted_bank_layout.h"
namespace tide::device_online {
struct WeightedBankPlan {
  at::Tensor permutation,offsets,buckets,row_connected,owner_connected;
  int64_t rows=0,banks=0;
};
// Stable O(R+B) device partition; roots are inspected only for a reverse plan.
WeightedBankPlan append_weighted_bank_plan(DeviceProgram&,const at::Tensor& owners,
  const at::Tensor& roots,int64_t banks,const at::Tensor& error,bool reverse);
void append_weighted_bank_snapshot(DeviceProgram&,const at::Tensor& connected,
  const at::Tensor& matrix,const at::Tensor& bias,const at::Tensor& saved_matrix,
  const at::Tensor& saved_bias,const at::Tensor& error,int64_t error_code=20);
// Shapes are static; each nonempty actual bank chooses its smallest bounded
// row bucket on device. Shared scratch is reused across mutually exclusive
// branches and banks. Each original row is scattered exactly once.
void append_weighted_bank_gemm(DeviceProgram&,const WeightedBankPlan&,
  const at::Tensor& input,const at::Tensor& matrix,const at::Tensor& bias,
  const at::Tensor& output,const at::Tensor& error);
void append_weighted_bank_gemm_vjp(DeviceProgram&,const WeightedBankPlan&,
  const at::Tensor& input,const at::Tensor& matrix,const at::Tensor& gradient,
  const at::Tensor& input_gradient,const at::Tensor& matrix_gradient,
  const at::Tensor& bias_gradient,const at::Tensor& error);
}
