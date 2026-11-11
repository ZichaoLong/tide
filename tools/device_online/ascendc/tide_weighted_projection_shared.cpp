#include "weighted_allocation_math.h"
// Single parameter bank: preserve actual use and saved values without R copies.
// GEMMs do the numerical projection/transpose; this kernel owns only masks,
// the forward bank snapshot and finite/structural-connection checks.
extern "C" __global__ __aicore__ void tide_weighted_projection_shared(GM_ADDR owners,GM_ADDR roots,
    GM_ADDR source_matrix,GM_ADDR source_bias,GM_ADDR matrix,GM_ADDR bias,GM_ADDR input,
    GM_ADDR connected,GM_ADDR error,int64_t rows,int64_t in,int64_t out,int64_t has_bias,int64_t mode){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  using namespace tide_weighted_device;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)owners);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  bool used=false;
  for(I r=0;r<rows;++r)if(((__gm__ I*)owners)[r]>=0&&(mode==0||((__gm__ uint8_t*)roots)[r]))used=true;
  bool good=true;
  for(I j=0;j<in*out;++j){
    const float value=used?(mode==0?((__gm__ float*)source_matrix)[j]:((__gm__ float*)matrix)[j]):0;
    ((__gm__ float*)matrix)[j]=value;if(!finite(value))good=false;
  }
  if(has_bias)for(I j=0;j<out;++j){
    const float value=used?(mode==0?((__gm__ float*)source_bias)[j]:((__gm__ float*)bias)[j]):0;
    ((__gm__ float*)bias)[j]=value;if(!finite(value))good=false;
  }
  if(mode==1){
    ((__gm__ uint8_t*)connected)[0]=used;
    for(I r=0;r<rows;++r)for(I j=0;j<in;++j){
      const float value=((__gm__ uint8_t*)roots)[r]?((__gm__ float*)input)[r*in+j]:0;
      ((__gm__ float*)input)[r*in+j]=value;if(!finite(value))good=false;
    }
  }
  if(!good)((__gm__ int32_t*)error)[0]=mode==0?20:23;
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
