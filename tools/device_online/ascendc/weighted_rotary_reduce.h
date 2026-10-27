#pragma once
#include "weighted_allocation_math.h"
namespace tide_weighted_rotary_device {
using namespace tide_weighted_device;
__aicore__ inline uint32_t word_at(uint32_t* digits,I bit){
  const I index=bit/32,shift=bit%32;uint32_t out=digits[index]>>shift;
  if(shift&&index<6)out|=digits[index+1]<<(32-shift);return out;
}
// Payne-Hanek reduction for a finite IEEE FP32 argument, without converting a
// huge quotient to a scalar int. floor((2/pi)*2^192), little-endian 32-bit words.
// A 24-bit significand gives a 216-bit product. For exponent <=127, the constant
// truncation is <2^-64 in quotient units; retain 64 fractional bits before RNE.
__aicore__ inline void reduce_one(float value,float& reduced,int32_t& quadrant){
  const uint32_t bits=AscendC::GetScalarBitcodeValue<float,uint32_t>(value),absolute=bits&0x7fffffff;
  if(absolute<=0x3f490fdbu){reduced=value;quadrant=0;return;}
  const uint32_t constant[6]={0x3c439041u,0xdb629599u,0xf534ddc0u,0xfc2757d1u,0x4e441529u,0xa2f9836eu};
  const uint32_t mantissa=(absolute&0x7fffff)|0x800000;const I exponent=I(absolute>>23)-127;
  uint32_t digits[7];uint64_t carry=0;
  for(I j=0;j<6;++j){const uint64_t product=uint64_t(constant[j])*mantissa+carry;digits[j]=uint32_t(product);carry=product>>32;}digits[6]=uint32_t(carry);
  const I point=215-exponent;uint32_t high=word_at(digits,point-32),low=word_at(digits,point-64);
  int32_t q=int32_t(word_at(digits,point)&3);const bool up=(high&0x80000000u)!=0;
  if(up){++q;low=~low+1;high=~high+(low==0?1u:0u);}
  // Only <=16-bit unsigned pieces enter scalar float conversion.
  const float fraction=float(int32_t(high>>16))*0x1p-16f+float(int32_t(high&0xffff))*0x1p-32f
    +float(int32_t(low>>16))*0x1p-48f+float(int32_t(low&0xffff))*0x1p-64f;
  float r=fraction*1.57079625129699707031f+fraction*7.54978941586159635335e-8f;
  if(up)r=-r;if(bits>>31){r=-r;q=-q;}reduced=r;quadrant=q&3;
}
__aicore__ inline void reduce(__gm__ float* input,__gm__ float* output,__gm__ int32_t* quadrant,
    __gm__ int32_t* error,I count){
  if(error[0])return;for(I i=0;i<count;++i){const float x=input[i];if(!finite(x)){error[0]=33;return;}
    float r;int32_t q;reduce_one(x,r,q);output[i]=r;quadrant[i]=q;}
}
__aicore__ inline void unfold(__gm__ float* cosine,__gm__ float* sine,__gm__ int32_t* quadrant,
    __gm__ int32_t* error,I count){
  if(error[0])return;for(I i=0;i<count;++i){const float c=cosine[i],s=sine[i];float a=c,b=s;
    switch(quadrant[i]){case 0:break;case 1:a=-s;b=c;break;case 2:a=-c;b=-s;break;case 3:a=s;b=-c;break;default:error[0]=33;return;}
    if(!finite(a)||!finite(b)){error[0]=33;return;}cosine[i]=a;sine[i]=b;}
}
}
