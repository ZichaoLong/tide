#include "weighted_allocation_math.h"
namespace {
using namespace tide_weighted_device;
__aicore__ inline bool sanitize(__gm__ float* p,I width,bool present){
  bool good=true;
  for(I j=0;j<width;++j){if(!present)p[j]=0;else if(!finite(p[j]))good=false;}
  return good;
}
}
extern "C" __global__ __aicore__ void tide_weighted_head_guard(GM_ADDR owners,GM_ADDR a,GM_ADDR b,
    GM_ADDR c,GM_ADDR d,GM_ADDR roots,GM_ADDR gradient,GM_ADDR guarded,GM_ADDR connected,
    GM_ADDR row_errors,GM_ADDR error,int64_t rows,int64_t aw,int64_t bw,int64_t cw,int64_t dw,int64_t stage){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)owners);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  for(I row=AscendC::GetBlockIdx();row<rows;row+=AscendC::GetBlockNum()){
    const bool present=((__gm__ I*)owners)[row]>=0;auto e=(__gm__ int32_t*)row_errors+row;*e=0;
    if(stage==2){
      const bool root=present&&((__gm__ uint8_t*)roots)[row];((__gm__ uint8_t*)connected)[row]=root;
      for(I j=0;j<aw;++j){const float x=root?((__gm__ float*)gradient)[row*aw+j]:0;
        if(!finite(x))*e=23;((__gm__ float*)guarded)[row*aw+j]=x;}
    }else{
      bool good=sanitize((__gm__ float*)a+row*aw,aw,present);
      if(stage==0){
        good=sanitize((__gm__ float*)b+row*bw,bw,present)&&good;
        good=sanitize((__gm__ float*)c+row*cw,cw,present)&&good;
        good=sanitize((__gm__ float*)d+row*dw,dw,present)&&good;
      }
      if(!good)*e=20;
    }
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
