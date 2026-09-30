#include "kernel_operator.h"
namespace {using I=int64_t;}
extern "C" __global__ __aicore__ void tide_journal(GM_ADDR incoming_meta,GM_ADDR incoming_values,
    GM_ADDR incoming_count,GM_ADDR metadata,GM_ADDR values,GM_ADDR count,GM_ADDR error,
    int64_t capacity,int64_t rows,int64_t columns,int64_t width,int64_t commit) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)metadata);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto status=(__gm__ int32_t*)error;auto n=(__gm__ I*)count;
  I add=((__gm__ I*)incoming_count)[0],old=n[0];
  if(status[0]==0) {
    if(add<0||add>rows||old<0||old>capacity)status[0]=2;
    else if(add>capacity-old)status[0]=1;
    else if(commit) {
      for(I i=0;i<add*columns;++i)((__gm__ I*)metadata)[old*columns+i]=((__gm__ I*)incoming_meta)[i];
      for(I i=0;i<add*width;++i)((__gm__ float*)values)[old*width+i]=((__gm__ float*)incoming_values)[i];
      n[0]=old+add;
    }
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
