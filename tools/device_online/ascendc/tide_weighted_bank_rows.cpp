#include "weighted_allocation_math.h"
// Gather/scatter one tile of an actual bank. A partial tail is structurally
// absent; it never reads a source/gradient or publishes an output row.
extern "C" __global__ __aicore__ void tide_weighted_bank_rows(GM_ADDR permutation,GM_ADDR offsets,
    GM_ADDR cursor,GM_ADDR source,GM_ADDR destination,GM_ADDR error,int64_t rows,int64_t banks,
    int64_t capacity,int64_t width,int64_t scatter,int64_t error_code){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  using namespace tide_weighted_device;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)offsets);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  const I bank=((__gm__ I*)cursor)[0],start=((__gm__ I*)cursor)[1];
  if(rows<1||banks<1||capacity<1||capacity>rows||width<1||bank<0||bank>=banks||start<0
      ||start%capacity||(scatter!=0&&scatter!=1)){((__gm__ int32_t*)error)[0]=2;return;}
  const I first=((__gm__ I*)offsets)[bank],end=((__gm__ I*)offsets)[bank+1];
  if(first<0||end<=first||end>rows||start>=end-first){((__gm__ int32_t*)error)[0]=2;return;}
  const I begin=first+start,count=end-begin;
  for(I j=AscendC::GetBlockIdx();j<capacity*width;j+=AscendC::GetBlockNum()){
    const I r=j/width,col=j%width;
    const I row=r<count?((__gm__ I*)permutation)[begin+r]:0;
    if(r<count&&(row<0||row>=rows)){((__gm__ int32_t*)error)[0]=2;return;}
    if(scatter){if(r<count){const float value=((__gm__ float*)source)[j];
      if(!finite(value))((__gm__ int32_t*)error)[0]=error_code;
      ((__gm__ float*)destination)[row*width+col]=value;}}
    else{const float value=r<count?((__gm__ float*)source)[row*width+col]:0;
      if(!finite(value))((__gm__ int32_t*)error)[0]=error_code;((__gm__ float*)destination)[j]=value;}
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
