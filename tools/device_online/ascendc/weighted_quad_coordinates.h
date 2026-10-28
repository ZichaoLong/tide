#pragma once
#include "weighted_allocation_math.h"
namespace tide_weighted_device {
// Exact ordered FP32 keys select the cap-weighted score quantile in 32 scans.
// This searches score values, not an approximate multiplier; no float stores
// an index or count. The returned pivot is an actual finite shifted score.
__aicore__ inline uint32_t ordered(float value){
  const auto bits=AscendC::GetScalarBitcodeValue<float,uint32_t>(value);
  return bits&0x80000000u?~bits:bits^0x80000000u;
}
__aicore__ inline float quad_pivot(__gm__ float* scores,__gm__ float* caps,I n,float shift,float target){
  uint64_t lo=0,hi=0xffffffffu;
  for(I iteration=0;iteration<32&&lo<hi;++iteration){
    const uint64_t mid=lo+(hi-lo+1)/2;float total=0;
    for(I j=0;j<n;++j)if(ordered(scores[j]-shift)>=mid)total+=caps[j];
    if(total>=target)lo=mid;else hi=mid-1;
  }
  const uint32_t code=static_cast<uint32_t>(lo),bits=code&0x80000000u?code^0x80000000u:~code;
  return AscendC::GetScalarBitcodeValue<uint32_t,float>(bits);
}
}
