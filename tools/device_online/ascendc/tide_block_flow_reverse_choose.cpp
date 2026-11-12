#include "block_flow_reverse_math.h"
extern "C" __global__ __aicore__ void tide_block_flow_reverse_choose(GM_ADDR journal,GM_ADDR ledger,GM_ADDR cursor,GM_ADDR branch,GM_ADDR error,
    int64_t regions,int64_t command){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);if(AscendC::GetBlockIdx()!=0)return;
  using I=int64_t;AscendC::GlobalTensor<int32_t> cache;cache.SetGlobalBuffer((__gm__ int32_t*)error);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  tide_block_flow_reverse_device::choose((__gm__ I*)journal,(__gm__ I*)ledger,(__gm__ I*)cursor,(__gm__ int32_t*)branch,(__gm__ int32_t*)error,regions,command);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
