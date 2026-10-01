#include "kernel_operator.h"
extern "C" __global__ __aicore__ void tide_state_reverse_flags(GM_ADDR decay,GM_ADDR retention,GM_ADDR decay_out,GM_ADDR retention_out,
    GM_ADDR error,int64_t samples,int64_t nodes) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<int32_t> cache;cache.SetGlobalBuffer((__gm__ int32_t*)error);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(!((__gm__ int32_t*)error)[0])for(int64_t n=0;n<nodes;++n)for(int64_t b=0;b<samples;++b) {
    ((__gm__ uint8_t*)decay_out)[n]|=((__gm__ uint8_t*)decay)[b*nodes+n];
    ((__gm__ uint8_t*)retention_out)[n]|=((__gm__ uint8_t*)retention)[b*nodes+n];
  }
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
