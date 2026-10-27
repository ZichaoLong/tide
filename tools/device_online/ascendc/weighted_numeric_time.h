#pragma once
#include "weighted_allocation_math.h"
namespace tide_weighted_time_device {
// Correct round-to-nearest-even conversion of nonnegative int64 metadata to
// FP32 numerical time. Never route through a 32-bit signed scalar conversion.
// Metadata itself remains int64; this is only for declared decay/position math.
__aicore__ inline float as_float(int64_t value){
  if(value==0)return 0;
  int64_t e=0,n=value;while(n>1){n>>=1;++e;}
  if(e<=23)return float(int32_t(value));
  const int64_t shift=e-23,half=int64_t(1)<<(shift-1),tail=value&((int64_t(1)<<shift)-1);
  int64_t significand=value>>shift;
  if(tail>half||(tail==half&&(significand&1)))++significand;
  if(significand==(int64_t(1)<<24)){significand>>=1;++e;}
  const uint32_t bits=uint32_t(e+127)<<23|uint32_t(significand&0x7fffff);
  return AscendC::GetScalarBitcodeValue<uint32_t,float>(bits);
}
}
