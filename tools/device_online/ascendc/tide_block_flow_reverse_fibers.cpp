#include "block_flow_reverse_math.h"
extern "C" __global__ __aicore__ void tide_block_flow_reverse_fibers(GM_ADDR mapping,GM_ADDR lengths,GM_ADDR gradient,GM_ADDR roots,GM_ADDR external,GM_ADDR external_roots,GM_ADDR messages,GM_ADDR message_roots,GM_ADDR error,
    int64_t rows,int64_t capacity,int64_t width,int64_t external_rows,int64_t message_rows){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);if(AscendC::GetBlockIdx()!=0)return;
  using I=int64_t;AscendC::GlobalTensor<int32_t> cache;cache.SetGlobalBuffer((__gm__ int32_t*)error);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  tide_block_flow_reverse_device::fibers((__gm__ I*)mapping,(__gm__ I*)lengths,(__gm__ float*)gradient,(__gm__ uint8_t*)roots,(__gm__ float*)external,(__gm__ uint8_t*)external_roots,(__gm__ float*)messages,(__gm__ uint8_t*)message_roots,(__gm__ int32_t*)error,rows,capacity,width,external_rows,message_rows);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
