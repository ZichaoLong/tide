#include "block_flow_reverse_math.h"
extern "C" __global__ __aicore__ void tide_block_flow_reverse_ports(GM_ADDR sources,GM_ADDR present,GM_ADDR mass,GM_ADDR mass_roots,GM_ADDR content,GM_ADDR content_roots,GM_ADDR output,GM_ADDR connected,GM_ADDR error,
    int64_t frames,int64_t ports,int64_t nodes,int64_t width){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);if(AscendC::GetBlockIdx()!=0)return;
  using I=int64_t;AscendC::GlobalTensor<int32_t> cache;cache.SetGlobalBuffer((__gm__ int32_t*)error);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  tide_block_flow_reverse_device::ports((__gm__ I*)sources,(__gm__ uint8_t*)present,(__gm__ float*)mass,(__gm__ uint8_t*)mass_roots,(__gm__ float*)content,(__gm__ uint8_t*)content_roots,(__gm__ float*)output,(__gm__ uint8_t*)connected,(__gm__ int32_t*)error,frames,ports,nodes,width);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
