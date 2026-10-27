#pragma once
#include "weighted_raw_update.h"
namespace tide::device_online {
struct WeightedConvParameters {at::Tensor weight,bias;}; // [banks,taps,features], [banks,features]
struct WeightedConvTape {
  WeightedRawLayout layout;
  WeightedConvParameters rows;
  at::Tensor owners,indices,old_integer,mapping,taps,output;
  int64_t banks=0,stride=0;
};
struct WeightedConvGradient {
  WeightedConvParameters parameters;
  at::Tensor current,raw_cache,row_connected,cache_connected,owner_connected;
};
// Owners -1 mark absent rows. Cache is the selected initial/old/proposal state;
// current tap is always unweighted z, even if cache contains theta -> w*z.
// Raw/current/parameters/output use one explicit FP32/FP16 dtype. Multiply,
// reduction, saved taps and VJP are FP32. FP16 rounds once after bias; overflow
// is refused. Casts have the declared identity first-order VJP.
WeightedConvTape append_weighted_convolution(DeviceProgram&,const WeightedRawLayout&,const WeightedRawState&,
  const at::Tensor& time,const at::Tensor& current,const at::Tensor& owners,const WeightedConvParameters&,
  int64_t stride,const at::Tensor& error,int64_t budget=256*1024*1024);
WeightedConvGradient append_weighted_convolution_vjp(DeviceProgram&,const WeightedConvTape&,
  const at::Tensor& gradient,const at::Tensor& roots,const at::Tensor& error,int64_t budget=256*1024*1024);
}
