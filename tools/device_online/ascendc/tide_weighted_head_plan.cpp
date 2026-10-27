#include "weighted_allocation_math.h"
// Only metadata/masking: numerical affine maps and SiLU use batched vendor ops.
extern "C" __global__ __aicore__ void tide_weighted_head_plan(GM_ADDR source,GM_ADDR owners,
    GM_ADDR input,GM_ADDR indices,GM_ADDR row_errors,GM_ADDR error,int64_t rows,int64_t width,int64_t banks){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  using namespace tide_weighted_device;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)owners);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  for(I row=AscendC::GetBlockIdx();row<rows;row+=AscendC::GetBlockNum()){
    const I owner=((__gm__ I*)owners)[row];auto e=(__gm__ int32_t*)row_errors+row;*e=0;
    (( __gm__ I*)indices)[row]=owner>=0&&owner<banks?owner:0;
    if(owner< -1||owner>=banks)*e=2;
    for(I j=0;j<width;++j){float x=0;
      if(owner>=0&&owner<banks){x=((__gm__ float*)source)[row*width+j];if(!finite(x))*e=20;}
      ((__gm__ float*)input)[row*width+j]=x;}
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
