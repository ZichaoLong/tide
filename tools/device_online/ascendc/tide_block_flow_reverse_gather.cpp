#include "block_flow_reverse_math.h"
extern "C" __global__ __aicore__ void tide_block_flow_reverse_gather(GM_ADDR nodes,GM_ADDR present,GM_ADDR incoming,GM_ADDR flags,GM_ADDR direct_mass,GM_ADDR direct_mass_roots,GM_ADDR direct_content,GM_ADDR direct_content_roots,GM_ADDR mass,GM_ADDR mass_roots,GM_ADDR content,GM_ADDR content_roots,GM_ADDR error,
    int64_t frames,int64_t count,int64_t all_nodes,int64_t width){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);if(AscendC::GetBlockIdx()!=0)return;
  using I=int64_t;AscendC::GlobalTensor<int32_t> cache;cache.SetGlobalBuffer((__gm__ int32_t*)error);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  tide_block_flow_reverse_device::gather((__gm__ I*)nodes,(__gm__ uint8_t*)present,(__gm__ float*)incoming,(__gm__ uint8_t*)flags,(__gm__ float*)direct_mass,(__gm__ uint8_t*)direct_mass_roots,(__gm__ float*)direct_content,(__gm__ uint8_t*)direct_content_roots,(__gm__ float*)mass,(__gm__ uint8_t*)mass_roots,(__gm__ float*)content,(__gm__ uint8_t*)content_roots,(__gm__ int32_t*)error,frames,count,all_nodes,width);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
